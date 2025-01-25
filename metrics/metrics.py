import torch
from abc import ABC, abstractmethod
from torch import nn
from utils.data_utils import mel_spectrogram
from utils.class_registry import ClassRegistry

metrics_registry = ClassRegistry()

class BaseMetric(ABC):
    @abstractmethod
    def __call__(self, val_iter, synthesize_wavs):
        pass

@metrics_registry.add_to_registry(name='l1_mel_diff')
class L1MelDiff(BaseMetric):
    def __init__(self, config):
        self.n_fft = config.mel.n_fft
        self.num_mels = config.mel.num_mels
        self.sampling_rate = config.mel.sampling_rate
        self.hop_size = config.mel.hop_size
        self.win_size = config.mel.win_size
        self.fmin = config.mel.fmin
        self.fmax = config.mel.fmax_for_loss
        self.device = config.exp.device
        self.l1_loss = nn.L1Loss()

    def __call__(self, real_batch, gen_batch):
        real_mel = real_batch['mel_for_loss'].to(self.device)
        gen_mel = mel_spectrogram(gen_batch['gen_wav'], self.n_fft, self.num_mels,
                                      self.sampling_rate, self.hop_size, self.win_size,
                                      self.fmin, self.fmax, center=False)
        return self.l1_loss(real_mel, gen_mel)
