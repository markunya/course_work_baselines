import torch
from utils.data_utils import mel_spectrogram
from utils.model_utils import requires_grad
from training.trainers.base_trainer import BaseTrainer

from training.trainers.base_trainer import gan_trainers_registry

@gan_trainers_registry.add_to_registry(name='hifipp_trainer')
class HifiPlusPlusTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self):
        pass
    
    def synthesize_wavs(self, batch):
        pass
