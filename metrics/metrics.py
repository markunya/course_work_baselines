import torch
import torchaudio
from abc import ABC, abstractmethod
import torchaudio.transforms as T
from pesq import pesq
import numpy as np
from pystoi import stoi
from torch import nn
from utils.data_utils import mel_spectrogram
from utils.class_registry import ClassRegistry
from collections import OrderedDict
from models.metric_models import Wav2Vec2MOS

metrics_registry = ClassRegistry()

class ResamleMetric(ABC):
    def __init__(self, config, target_sr):
        self.target_sr = target_sr
        self.device = config.exp.device
        self.resampler = T.Resample(orig_freq=config.mel.sampling_rate,
                                new_freq=self.target_sr).to(self.device)
        
    def resample(self, wav):
        return self.resampler(wav.unsqueeze(0)).squeeze(0)
    
    @abstractmethod
    def __call__(self, real_batch, gen_batch):
        pass

@metrics_registry.add_to_registry(name='l1_mel_diff')
class L1MelDiff:
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

@metrics_registry.add_to_registry(name='wb_pesq')
class WBPesq(ResamleMetric):
    def __init__(self, config):
        super().__init__(config, 16000)

    def __call__(self, real_batch, gen_batch):
        scores = []
        for real_wav, gen_wav in zip(real_batch['wav'], gen_batch['gen_wav']):
            real_wav_resampled = self.resample(real_wav.to(self.device))
            gen_wav_resampled = self.resample(gen_wav.to(self.device))

            real_wav_np = real_wav_resampled.cpu().numpy()
            gen_wav_np = gen_wav_resampled.cpu().numpy()
            try:
                pesq_score = pesq(self.target_sr, real_wav_np, gen_wav_np, 'wb')
            except:
                pesq_score = 1
            scores.append(pesq_score)

        return np.mean(scores)
    
@metrics_registry.add_to_registry(name='stoi')
class STOI:
    def __init__(self, config):
        self.sr = config.mel.sampling_rate

    def __call__(self, real_batch, gen_batch):
        scores = []
        for real_wav, gen_wav in zip(real_batch['wav'], gen_batch['gen_wav']):
            real_wav_np = real_wav.cpu().numpy()
            gen_wav_np = gen_wav.cpu().numpy()

            stoi_score = stoi(real_wav_np, gen_wav_np, self.sr, extended=True)
            scores.append(stoi_score)

        return np.mean(scores)

@metrics_registry.add_to_registry(name='si_sdr')
class SISDR:
    def __init__(self, config):
        self.device = config.exp.device

    def __call__(self, real_batch, gen_batch):
        real_wavs = real_batch['wav'].to(self.device).squeeze(1)
        gen_wavs = gen_batch['gen_wav'].to(self.device).squeeze(1)

        alpha = (gen_wavs * real_wavs).sum(
            dim=1, keepdim=True
        ) / real_wavs.square().sum(dim=1, keepdim=True)
        real_wavs_scaled = alpha * real_wavs
        e_target = real_wavs_scaled.square().sum(dim=1)
        e_res = (gen_wavs - real_wavs).square().sum(dim=1)
        si_sdr = 10 * torch.log10(e_target / e_res).cpu().numpy()

        return np.mean(si_sdr)   

def extract_prefix(prefix, weights):
    result = OrderedDict()
    for key in weights:
        if key.find(prefix) == 0:
            result[key[len(prefix) :]] = weights[key]
    return result


@metrics_registry.add_to_registry(name="mosnet")
class MOSNet(ResamleMetric):
    def __init__(self, config):
        self.model = Wav2Vec2MOS("metrics/weights/wave2vec2mos.pth").to(config.exp.device)
        super().__init__(config, self.model.sample_rate)

    def __call__(self, real_batch, gen_batch):
        mos_scores = []

        for gen_wav in gen_batch['gen_wav']:
            gen_wav = gen_wav / gen_wav.abs().max()
            gen_wav_resampled = self.resample(gen_wav)

            input_values = self.model.processor(
                gen_wav_resampled.cpu().numpy(), return_tensors="pt", sampling_rate=self.model.sample_rate
            ).input_values.to("cuda")

            with torch.no_grad():
                mos_score = self.model(input_values).item()
            mos_scores.append(mos_score)

        return np.mean(mos_scores)
