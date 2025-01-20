import os
import torch

from abc import abstractmethod
from datasets.dataloaders import InfiniteLoader
from training.loggers import TrainingLogger
from training.losses.losses import LossBuilder
from torch.utils.data import DataLoader

from models.hifigan_models import models_registry
from training.optimizers import optimizers_registry
from training.schedulers import schedulers_registry
from datasets.datasets import datasets_registry
from metrics.metrics import metrics_registry

from utils.data_utils import read_file_list

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
        self.inference_out_dir = os.path.join(self.exp_dir, 'inference_out')
        self.checkpoints_dir = os.path.join(self.exp_dir, 'checkpoints')
        print('Директории для эксперимента успешно инициализированы')

    def _create_model(self, model_name, model_config):
        model_class = models_registry[model_name]
        return model_class(**model_config['args']).to(self.device)
    
    def setup_models(self):
        self.models = {}
        for model_name, model_config in self.config.models.items():
            self.models[model_name] = self._create_model(model_name, model_config)
        print(f'Модели успешно инициализированы')

    def _create_optimizer(self, model_name, model_config):
        optimizer_name = model_config['optimizer']['name']
        optimizer_class = optimizers_registry[optimizer_name]
        return optimizer_class(self.models[model_name].parameters(), **model_config['optimizer']['args'])
    
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
        print(f'Оптимизаторы и шедулеры успешно настроены: {list(self.optimizers.keys())}')

    def setup_losses(self):
        self.loss_builders = {}
        for model_name, model_config in self.config.models.items():
            self.loss_builders[model_name] = LossBuilder(model_config)
        print(f'Функции потерь успешно настроены для каждой модели')

    def setup_metrics(self):
        self.metrics = []
        for metric_name in self.config.train.val_metrics:
            metric = metrics_registry[metric_name]()
            self.metrics.append((metric_name, metric))
        print('Метрики успешно инициализированы')

    def setup_logger(self):
        self.logger = TrainingLogger(self.config)
        if self.config.exp.use_wandb:
            print('Логгер успешно инициализирован')

    def setup_datasets(self):
        train_files_list = read_file_list(self.config.data.train_data_file_path)
        self.train_dataset = datasets_registry[self.config.data.dataset](
                files_list=train_files_list,
                config=self.config
            )

        val_files_list = read_file_list(self.config.data.val_data_file_path)
        self.val_dataset = datasets_registry[self.config.data.dataset](
                files_list=val_files_list,
                config=self.config
        )
        print('Датасеты для обучения и валидации успешно инициализированы')

    
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
            batch_size=self.config.data.val_batch_size,
            shuffle=False,
            num_workers=self.config.data.workers,
            pin_memory=True,
        )

    def setup_dataloaders(self):
        self.setup_train_dataloader()
        self.setup_val_dataloader()
        print('Загрузчики данных успешно инициализированны')

    def to_train(self):
        for model in self.models.values():
            model.train()

    def to_eval(self):
        for model in self.models.values():
            model.eval()

    def training_loop(self):
        self.to_train()

        for self.step in range(self.start_step, self.config.train.steps + 1):
            losses_dict = self.train_step()
            self.logger.update_losses(losses_dict)

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
                path = os.path.join(self.checkpoints_dir, f'{model_name}_checkpoint_{self.step}.pth')
                torch.save(checkpoint, path)
                print(f'Чекпоинт для модели {model_name} на шаге {self.step} сохранён по пути {path}')
            except Exception as e:
                print(f'Не удалось сохранить чекпоинт для модели {model_name} на шаге {self.step}: {e}')

    @torch.no_grad()
    def validate(self):
        self.to_eval()
        metrics_dict = {}

        batch = next(iter(self.val_dataloader))
        synthesized_wavs = self.synthesize_wavs(batch)

        for metric_name, metric in self.metrics:
            val_iter = iter(self.val_dataloader)
            metrics_dict[metric_name] = metric(val_iter, self.synthesize_wavs)

        self.logger.log_dict_of_wavs(synthesized_wavs, step=self.step)
        self.logger.log_val_metrics(metrics_dict)
        print(f"Validation completed. Metrics: {metrics_dict}")
    
    @torch.no_grad()
    def inference(self):
        raise NotImplementedError('Will be implemented later')
    
    @abstractmethod
    def synthesize_wavs(self):
        pass

    @abstractmethod
    def train_step(self):
        pass
