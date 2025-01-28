import os
import torch

from abc import abstractmethod
from datasets.dataloaders import InfiniteLoader
from training.loggers import TrainingLogger
from training.losses.losses import LossBuilder
from torch.utils.data import DataLoader
from training.schedulers import ReduceLrOnEach
from tqdm import tqdm
from utils.class_registry import ClassRegistry

from models.hifigan_models import models_registry
from training.optimizers import optimizers_registry
from training.schedulers import schedulers_registry
from datasets.datasets import datasets_registry
from metrics.metrics import metrics_registry

from utils.data_utils import read_file_list

gan_trainers_registry = ClassRegistry()

class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.exp.device
        self.start_step = config.train.start_step
        self.step = self.start_step

    def setup_training(self):
        self.setup_experiment_dir()

        self.setup_models()
        self.setup_optimizers()
        self.setup_losses()

        self.setup_metrics()
        self.setup_logger()

        self.setup_datasets()
        self.setup_dataloaders()


    def setup_inference(self):
        self.setup_experiment_dir()

        self.setup_models()

        self.setup_metrics()
        self.setup_logger()

        self.setup_datasets()
        self.setup_dataloaders()

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

    def _create_model(self, model_name, model_config):
        model_class = models_registry[model_name]
        model = model_class(**model_config['args']).to(self.device)

        checkpoint_path = model_config.get('checkpoint_path')
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            print(f'Loading checkpoint for {model_name} from {checkpoint_path}...')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            if checkpoint_path:
                print(f'Warning: Checkpoint not found at {checkpoint_path}. Initializing {model_name} from scratch.')
            else:
                print(f'No checkpoint specified for {model_name}. Initializing from scratch.')

        return model
    
    def setup_models(self):
        self.models = {}
        for model_name, model_config in self.config.models.items():
            self.models[model_name] = self._create_model(model_name, model_config)
        tqdm.write(f'Models successfully initialized: {list(self.models.keys())}')

    def _create_optimizer(self, model_name, model_config):
        optimizer_name = model_config['optimizer']['name']
        optimizer_class = optimizers_registry[optimizer_name]
        optimizer = optimizer_class(self.models[model_name].parameters(), **model_config['optimizer']['args'])

        checkpoint_path = model_config.get('checkpoint_path')
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            print(f'Loading optimizer state for {model_name} from {checkpoint_path}...')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                print(f'Warning: optimizer_state_dict not found in {checkpoint_path}. Starting fresh optimizer for {model_name}.')

        return optimizer
    
    def _create_scheduler(self, model_name, model_config):
        scheduler_name = model_config['scheduler']['name']
        scheduler_class = schedulers_registry[scheduler_name]
        return scheduler_class(self.optimizers[model_name], **model_config['scheduler']['args'])

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
            self.loss_builders[model_name] = LossBuilder(model_config)
        tqdm.write(f'Loss functions successfully initialized')

    def setup_metrics(self):
        self.metrics = []
        for metric_name in self.config.train.val_metrics:
            metric = metrics_registry[metric_name](self.config)
            self.metrics.append((metric_name, metric))
        tqdm.write('Metrics successfully initialized')

    def setup_logger(self):
        self.logger = TrainingLogger(self.config)
        if self.config.exp.use_wandb:
            tqdm.write('Logger successfully initialized')

    def setup_datasets(self):
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
                **self.config.data.dataset_args
        )
        tqdm.write('Datasets for train and validation successfully initialized')

    
    def setup_train_dataloader(self):
        self.train_dataloader = InfiniteLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
            num_workers=self.config.data.workers,
            pin_memory=True,
        )

    def setup_val_dataloader(self):
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.data.workers,
            pin_memory=True,
        )

    def setup_dataloaders(self):
        self.setup_train_dataloader()
        self.setup_val_dataloader()
        tqdm.write('Dataloaders successfully initialized')

    def to_train(self):
        for model in self.models.values():
            model.train()

    def to_eval(self):
        for model in self.models.values():
            model.eval()

    def training_loop(self):
        self.to_train()

        with tqdm(total=self.config.train.steps, desc='Training Progress', unit='step') as progress:
            for self.step in range(self.start_step, self.config.train.steps + 1):
                losses_dict = self.train_step()
                
                for scheduler_name in self.schedulers:
                    if self.schedulers[scheduler_name].reduce_time == ReduceLrOnEach.step:
                        self.schedulers[scheduler_name].step()

                if self.step % len(self.train_dataloader) == 0:
                    for scheduler_name in self.schedulers:
                        if self.schedulers[scheduler_name].reduce_time == ReduceLrOnEach.epoch:
                            self.schedulers[scheduler_name].step()

                self.logger.update_losses(losses_dict)
                progress.set_postfix({
                    model_name + '_lr': optimizer.param_groups[0]['lr']
                    for model_name, optimizer in self.optimizers.items()
                })
                progress.update(1)

                if self.step % self.config.train.val_step == 0:
                    self.validate()

                if self.step % self.config.train.log_step == 0:
                    self.logger.log_train_losses(self.step)

                if self.step % self.config.train.checkpoint_step == 0:
                    self.save_checkpoint()

    
    def save_checkpoint(self):
        for model_name in self.models.keys():
            try:
                checkpoint = {
                    'state_dict': self.models[model_name].state_dict(),
                    'optimizer_state_dict': self.optimizers[model_name].state_dict()
                }
                path = os.path.join(self.checkpoints_dir,
                                f'{model_name}_checkpoint_{self.step}_{self.config.exp.run_name}.pth')
                torch.save(checkpoint, path)
                tqdm.write(f'Checkpoint for {model_name} on step {self.step} saved to {path}')
            except Exception as e:
                tqdm.write(f'An error occured when saving checkpoint for {model_name} on step {self.step}: {e}')

    @torch.no_grad()
    def validate(self):
        self.to_eval()
        metrics_dict = {}
        for batch in self.val_dataloader:
            for metric_name, metric in self.metrics:
                metrics_dict['val_' + metric_name] = metric(batch, self.synthesize_wavs(batch))

        self.logger.log_val_metrics(metrics_dict, self.step)

        iterator = iter(self.val_dataloader)
        for _ in range(self.config.exp.log_batch_size):
            batch = next(iterator)
            gen_batch = self.synthesize_wavs(batch)
            self.logger.log_synthesized_batch(gen_batch, self.config.mel.sampling_rate, step=self.step)

        tqdm.write('Validation completed.' + 
                (f'Metrics: {metrics_dict}' if len(metrics_dict) > 0 else ''))
    
    @torch.no_grad()
    def inference(self):
        raise NotImplementedError('Will be implemented later')
    
    @abstractmethod
    def synthesize_wavs(self, batch):
        pass

    @abstractmethod
    def train_step(self):
        pass
