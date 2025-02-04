import wandb
import omegaconf
import os
from collections import defaultdict

class WandbLogger:
    def __init__(self, config):
        wandb.login(key=os.environ['WANDB_KEY'].strip())
        self.wandb_args = {
            "id": wandb.util.generate_id(),
            "project": config.exp.project_name,
            "name": config.exp.run_name,
            "config": config,
        }

        wandb.init(**self.wandb_args, resume="allow")
        config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
        wandb.config.update(config_dict)


    @staticmethod
    def log_values(values_dict: dict, step: int):
        wandb.log(values_dict, step=step)

    @staticmethod
    def log_wavs(wavs_dict: dict, sample_rate: float, step: int):
        for name, wav in wavs_dict.items():
            wandb.log({name: wandb.Audio(wav.flatten(), sample_rate)}, step=step)

def log_if_enabled(func):
    def wrapper(self, *args, **kwargs):
        if self.use_logger:
            return func(self, *args, **kwargs)
    return wrapper

class TrainingLogger:
    def __init__(self, config):
        self.use_logger = config.exp.use_wandb
        if not self.use_logger:
            return
        self.logger = WandbLogger(config)
        self.losses_memory = defaultdict(list)

    @log_if_enabled
    def log_train_losses(self, step: int):
        averaged_losses = {name: sum(values) / len(values) for name, values in self.losses_memory.items()}
        self.logger.log_values(averaged_losses, step)
        self.losses_memory.clear()

    @log_if_enabled
    def log_metrics(self, metrics: dict, step: int):
        self.logger.log_values(metrics, step)

    @log_if_enabled
    def log_synthesized_batch(self, gen_batch, sample_rate, step):
        wavs_dict = {}
        for name, gen_wav in zip(gen_batch['name'], 
                                gen_batch['gen_wav']):
            wavs_dict[name] = gen_wav.cpu()
        self.logger.log_wavs(wavs_dict, sample_rate, step)

    @log_if_enabled
    def update_losses(self, losses_dict):
        for loss_name, loss_val in losses_dict.items():
            self.losses_memory[loss_name].append(loss_val)
