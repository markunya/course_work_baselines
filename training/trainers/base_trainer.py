import os
import torch
import numpy as np
import torch.nn as nn
from autoclip.torch import QuantileClip

from scipy import stats
from abc import abstractmethod
from datasets.dataloaders import InfiniteLoader
from training.loggers import TrainingLogger
from training.losses.loss_builder import LossBuilder
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.class_registry import ClassRegistry

from models import models_registry
from training.optimizers import optimizers_registry
from training.schedulers import schedulers_registry
from datasets.datasets import datasets_registry
from metrics.metrics import metrics_registry

from utils.data_utils import save_wavs_to_dir
from utils.model_utils import unwrap_model

gan_trainers_registry = ClassRegistry()

class BaseTrainerHelpers:
    def __init__(self, config):
        self.config = config
        self.device = config.exp.device
        self.step=1
        if 'train' in config:
            self.start_step = config.train.start_step
            self.step = self.start_step
        self.multi_gpu = False
    
    def _create_model(self, model_name, model_config):
        model_class = models_registry[model_name]
        model = model_class(**model_config['args'])
        
        checkpoint_path = model_config.get('checkpoint_path')
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            tqdm.write(f'Loading checkpoint for {model_name} from {checkpoint_path}...')
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            if checkpoint_path:
                tqdm.write(f'Warning: Checkpoint not found at {checkpoint_path}. Initializing {model_name} from scratch.')
            else:
                tqdm.write(f'No checkpoint specified for {model_name}. Initializing from scratch.')

        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            if not self.multi_gpu:
                tqdm.write(f'Will be used {n_gpus} GPUs')
            self.multi_gpu = True
            model = nn.DataParallel(model)

        model = model.to(self.device)

        return model

    def _create_optimizer(self, model_name, model_config):
        optimizer_name = model_config['optimizer']['name']
        optimizer_class = optimizers_registry[optimizer_name]
        optimizer = optimizer_class(self.models[model_name].parameters(), **model_config['optimizer']['args'])

        load_from_checkpoint = True
        if 'load_optimizer_from_checkpoint' in model_config \
            and not model_config.load_optimizer_from_checkpoint:
            tqdm.write(
                f'load_optimizer_from_checkpoint for model {model_name} set to false, ' +
                'initializing optimizer from scratch'
            )
            load_from_checkpoint = False
        
        if load_from_checkpoint:
            checkpoint_path = model_config.get('checkpoint_path')

            if checkpoint_path is not None and os.path.isfile(checkpoint_path):
                tqdm.write(f'Loading optimizer state for {model_name} from {checkpoint_path}...')
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except Exception as e:
                        tqdm.write(f'An error occured when loading checkpoint for optimizer for {model_name}: {e}')
                else:
                    tqdm.write(f'Warning: optimizer_state_dict not found in {checkpoint_path}. '
                               + 'Starting fresh optimizer for {model_name}.')

        clip_args = {
            'quantile': 1.0
        }
        
        if 'clip_quantile' in self.config.train:
            clip_args['quantile'] = self.config.train.clip_quantile
            tqdm.write(f'Clip quantile set to {clip_args['quantile']} for {model_name}')
        if 'clip_history' in self.config.train:
            clip_args['history'] = self.config.train.clip_history
            tqdm.write(f'Clip history set to {clip_args['history']} for {model_name}')

        return QuantileClip.as_optimizer(
                    optimizer=optimizer,
                    global_threshold=True,
                    **clip_args
                )

    def _create_scheduler(self, model_name, model_config):
        scheduler_name = model_config['scheduler']['name']
        scheduler_class = schedulers_registry[scheduler_name]
        return scheduler_class(self.optimizers[model_name], **model_config['scheduler']['args'])
    
    def _compute_metrics(self, batch, gen_batch, metrics_dict, action):
        for metric_name, metric in self.metrics:
            value = metric(batch, gen_batch)
            assert isinstance(value, float), f"Each metric result must be float type, but {metric_name} returned {type(value)}"

            key = f'{action}_{metric_name}'
            if key not in metrics_dict:
                metrics_dict[key] = []

            metrics_dict[key].append(value)
    
    def _avg_computed_metrics(self, metrics_dict, action):
        ci_dict = {}
        
        for metric_name, _ in self.metrics:
            key = f"{action}_{metric_name}"
            values = np.array(metrics_dict[key])

            mean = np.mean(values)
            metrics_dict[key] = float(mean)

            if len(values) > 1:
                std = np.std(values, ddof=1)
                sem = std / np.sqrt(len(values))
                t = stats.t.ppf(0.975, df=len(values)-1)
                u = t * sem
            else:
                u = 0.0

            ci_dict[key] = float(u)

        return ci_dict

    def _log_synthesized_batch(self, iterator):
        for _ in range(self.config.exp.log_batch_size):
            batch = next(iterator)
            gen_batch = self.synthesize_wavs(batch)
            sr = self.config.mel.out_sr if 'out_sr' in self.config.mel else self.config.mel.in_sr
            self.logger.log_synthesized_batch(gen_batch, sr, step=self.step)

    def _print_metrics(self, metrics_dict, ci_dict):
        for key in metrics_dict.keys():
            mean = metrics_dict[key]
            u = ci_dict[key]
            tqdm.write(f"{key}: {mean:.4f} ± {u:.4f}")

class BaseTrainer(BaseTrainerHelpers):
    def __init__(self, config):
        super().__init__(config)

    def setup_training(self):
        self.setup_logger()
        
        self.setup_experiment_dir()

        self.setup_models()
        self.setup_optimizers()
        self.setup_losses()

        self.setup_val_metrics()

        if 'dataset' in self.config.data:
            self.setup_trainval_datasets()
        elif 'train_dataset' in self.config.data and 'val_dataset' in self.config.data:
            self.setup_train_dataset()
            self.setup_val_dataset()
        else:
            raise ValueError("Invalid config: data section must contain either dataset"
                             "or train_dataset and val_dataset")
        
        self.setup_trainval_dataloaders()


    def setup_inference(self):
        self.setup_experiment_dir()

        self.setup_models()

        self.setup_inf_metrics()
        self.setup_logger()

        self.setup_inference_dataset()
        self.setup_inference_dataloader()

    def setup_experiment_dir(self):
        self.exp_dir = self.config.exp.exp_dir

        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
            tqdm.write(f'Experiment directory \'{self.exp_dir}\' created.')

        self.inference_out_dir = os.path.join(self.exp_dir, 'inference_out')
        self.checkpoints_dir = os.path.join(self.exp_dir, 'checkpoints')

        for dir_path in [self.inference_out_dir, self.checkpoints_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                tqdm.write(f'Subdirectory \'{dir_path}\' created.')

        tqdm.write('Experiment dir successfully initialized')
    
    def setup_models(self):
        self.models = {}
        for model_name, model_config in self.config.models.items():
            self.models[model_name] = self._create_model(model_name, model_config)
        tqdm.write(f'Models successfully initialized: {list(self.models.keys())}')

    def setup_optimizers(self):
        self.optimizers = {}
        self.schedulers = {}
        for model_name, model_config in self.config.models.items():
            self.optimizers[model_name] = self._create_optimizer(model_name, model_config)
            self.schedulers[model_name] = self._create_scheduler(model_name, model_config)
        tqdm.write(f'Optimizers and schedulers successfully initialized')

    def setup_losses(self):
        self.loss_builders = {}
        for model_name, model_config in self.config.models.items():
            self.loss_builders[model_name] = LossBuilder(self.device, model_config['losses'])
        tqdm.write(f'Loss functions successfully initialized')

    def setup_val_metrics(self):
        self.metrics = []
        for metric_name in self.config.train.val_metrics:
            metric = metrics_registry[metric_name](self.config)
            self.metrics.append((metric_name, metric))
        tqdm.write('Validation metrics successfully initialized')

    def setup_inf_metrics(self):
        self.metrics = []
        for metric_name in self.config.inference.metrics:
            metric = metrics_registry[metric_name](self.config)
            self.metrics.append((metric_name, metric))
        tqdm.write('Inference metrics successfully initialized')

    def setup_logger(self):
        self.logger = TrainingLogger(self.config)
        if self.config.exp.use_wandb:
            tqdm.write('Logger successfully initialized')

    def setup_train_dataset(self):
        self.train_dataset = datasets_registry[self.config.data.train_dataset](
                self.config.data.train_data_root,
                self.config.data.train_data_file_path,
                self.config.mel,
                **self.config.data.train_dataset_args
            )
        tqdm.write('Train dataset successfully initialized')
        tqdm.write(f'Will be used {self.config.data.train_dataset} for training')

    def setup_val_dataset(self):
        self.val_dataset = datasets_registry[self.config.data.val_dataset](
                self.config.data.val_data_root,
                self.config.data.val_data_file_path,
                self.config.mel,
                split=False,
                eval=True,
                **self.config.data.val_dataset_args
        )
        tqdm.write('Validation dataset successfully initialized')
        tqdm.write(f'Will be used {self.config.data.val_dataset} for validating')

    def setup_trainval_datasets(self):
        self.train_dataset = datasets_registry[self.config.data.dataset](
                self.config.data.trainval_data_root,
                self.config.data.train_data_file_path,
                self.config.mel,
                **self.config.data.dataset_args
            )

        self.val_dataset = datasets_registry[self.config.data.dataset](
                self.config.data.trainval_data_root,
                self.config.data.val_data_file_path,
                self.config.mel,
                split=False,
                eval=True,
                **self.config.data.dataset_args
        )
        tqdm.write('Datasets for train and validation successfully initialized')
        tqdm.write(f'Will be used {self.config.data.dataset} both for training and validating')

    def setup_inference_dataset(self):
        self.inference_dataset = datasets_registry[self.config.data.dataset](
                self.config.data.inference_data_root,
                self.config.data.inference_data_file_path,
                self.config.mel,
                split=False,
                eval=True,
                **self.config.data.dataset_args
            )
        tqdm.write('Dataset for inference successfully initialized')
    
    def setup_train_dataloader(self):
        self.train_dataloader = InfiniteLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.data.workers,
            pin_memory=True,
        )
        tqdm.write('Train dataloader successfully initialized')

    def setup_val_dataloader(self):
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.data.workers,
            pin_memory=True,
        )
        tqdm.write('Val dataloader successfully initialized')

    def setup_inference_dataloader(self):
        self.inference_dataloader = DataLoader(
            self.inference_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.data.workers,
            pin_memory=True,
        )
        tqdm.write('Inference dataloader successfully initialized')

    def setup_trainval_dataloaders(self):
        self.setup_train_dataloader()
        self.setup_val_dataloader()

    def to_train(self):
        for model in self.models.values():
            model.train()

    def to_eval(self):
        for model in self.models.values():
            model.eval()

    def step_schedulers(self):
        for scheduler in self.schedulers.values():
            step = (scheduler.reduce_time == 'step')
            epoch = (scheduler.reduce_time == 'epoch') \
                        and (self.step % len(self.train_dataloader) == 0)
            period = (scheduler.reduce_time == 'period') \
                        and (self.step % scheduler.period == 0)
            if step or epoch or period:
                scheduler.step()

    def training_loop(self):
        with tqdm(total=self.config.train.steps, desc='Training Progress', unit='step') as progress:
            progress.update(self.step - 1)
            for self.step in range(self.start_step, self.config.train.steps + 1):
                self.to_train()
                
                losses_dict = self.train_step()
                self.step_schedulers()

                self.logger.update_losses(losses_dict)
                progress.set_postfix({
                    model_name + '_lr': optimizer.param_groups[0]['lr']
                    for model_name, optimizer in self.optimizers.items()
                })
                progress.update(1)

                if self.step == self.start_step:
                    iterator = iter(self.val_dataloader)
                    for _ in range(self.config.exp.log_batch_size):
                        batch = next(iterator)
                        input_batch = {
                            'gen_wav': batch['input_wav'] if 'input_wav' in batch else batch['wav'],
                            'name': batch['name']
                        }
               
                        sr = self.config.mel.in_sr
                        self.logger.log_synthesized_batch(input_batch, sr, step=self.step)

                if self.step % self.config.train.checkpoint_step == 0:
                    self.save_checkpoint()

                if self.step % self.config.train.val_step == 0:
                    self.validate()

                if self.step % self.config.train.log_step == 0:
                    self.logger.log_train_losses(self.step)

    
    def save_checkpoint(self):
        for model_name in self.models.keys():
            try:
                dir_path = os.path.join(self.checkpoints_dir, self.config.exp.run_name, model_name)
                os.makedirs(dir_path, exist_ok=True)

                checkpoint = {
                    'state_dict': unwrap_model(self.models[model_name]).state_dict(),
                    'optimizer_state_dict': self.optimizers[model_name].state_dict()
                }

                path = os.path.join(dir_path, f'{model_name}_checkpoint_{self.step}_{self.config.exp.run_name}.pth')
                torch.save(checkpoint, path)

                tqdm.write(f'Checkpoint for {model_name} on step {self.step} saved to {path}')
            except Exception as e:
                tqdm.write(f'An error occured when saving checkpoint for {model_name} on step {self.step}: {e}')

    @torch.no_grad()
    def validate(self):
        self.to_eval()
        metrics_dict = {}

        for batch in tqdm(self.val_dataloader, desc=f"Validating Progress on {self.step}"):
            gen_batch = self.synthesize_wavs(batch)
            self._compute_metrics(batch, gen_batch, metrics_dict, action='val')

        ci_dict = self._avg_computed_metrics(metrics_dict, action='val')

        self.logger.log_metrics(metrics_dict, self.step)
        self._log_synthesized_batch(iter(self.val_dataloader))

        tqdm.write('Validation completed:')
        self._print_metrics(metrics_dict, ci_dict)

    @torch.no_grad()
    def inference(self):
        self.to_eval()
        metrics_dict = {}

        for batch in tqdm(self.inference_dataloader, desc=f"Inference Progress"):
            gen_batch = self.synthesize_wavs(batch)
            run_inf_dir = os.path.join(self.inference_out_dir, self.config.exp.run_name)

            in_sr = self.config.mel.in_sr
            out_sr = self.config.mel.out_sr if 'out_sr' in self.config.mel else in_sr
                
            if self.config.inference.save_samples:
                if 'input_wav' in batch:
                    save_wavs_to_dir(batch['input_wav'], batch['name'],
                                    os.path.join(run_inf_dir, 'input'), in_sr)
                if 'wav' in batch:
                    save_wavs_to_dir(batch['wav'], batch['name'],
                                    os.path.join(run_inf_dir, 'ground_truth'), out_sr)
                save_wavs_to_dir(gen_batch['gen_wav'], gen_batch['name'],
                                    os.path.join(run_inf_dir, 'generated'), out_sr)

            self._compute_metrics(batch, gen_batch, metrics_dict, action='inf')
        
        ci_dict = self._avg_computed_metrics(metrics_dict, action='inf')

        self.logger.log_metrics(metrics_dict, 0)
        self._log_synthesized_batch(iter(self.inference_dataloader))

        tqdm.write('Inference completed:')
        self._print_metrics(metrics_dict, ci_dict)
    
    @abstractmethod
    def synthesize_wavs(self, batch):
        pass

    @abstractmethod
    def train_step(self):
        pass
