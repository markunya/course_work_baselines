import torch
import torchaudio
import torchaudio.transforms as T
from pesq import pesq
import numpy as np
from pystoi import stoi
from torch import nn
from utils.data_utils import mel_spectrogram
from utils.class_registry import ClassRegistry
from collections import OrderedDict
from transformers import Wav2Vec2Model, Wav2Vec2Processor

metrics_registry = ClassRegistry()


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
class WBPesq:
    def __init__(self, config, target_sr=16000):
        self.target_sr = target_sr
        self.device = config.exp.device
        self.resampler = T.Resample(orig_freq=config.mel.sampling_rate, new_freq=self.target_sr).to(self.device)

    def resample(self, wav):
        return self.resampler(wav.unsqueeze(0)).squeeze(0)

    def __call__(self, real_batch, gen_batch):
        total_pesq = 0.0
        batch_size = len(real_batch['wav'])

        for i in range(batch_size):
            real_wav = real_batch['wav'][i].to(self.device)
            gen_wav = gen_batch['gen_wav'][i]

            real_wav_resampled = self.resample(real_wav)
            gen_wav_resampled = self.resample(gen_wav)

            real_wav_np = real_wav_resampled.cpu().numpy()
            gen_wav_np = gen_wav_resampled.cpu().numpy()

            pesq_score = pesq(self.target_sr, real_wav_np, gen_wav_np, 'wb')
            total_pesq += pesq_score

        return total_pesq / batch_size
    
@metrics_registry.add_to_registry(name='stoi')
class STOI:
    def __init__(self, config, target_sr=10000):
        self.target_sr = target_sr
        self.device = config.exp.device
        self.resampler = T.Resample(orig_freq=config.mel.sampling_rate, new_freq=self.target_sr).to(self.device)

    def resample(self, wav):
        return self.resampler(wav.unsqueeze(0)).squeeze(0)

    def __call__(self, real_batch, gen_batch):
        total_stoi = 0.0
        batch_size = len(real_batch['wav'])

        for i in range(batch_size):
            real_wav = real_batch['wav'][i].to(self.device)
            gen_wav = gen_batch['gen_wav'][i]

            real_wav_resampled = self.resample(real_wav)
            gen_wav_resampled = self.resample(gen_wav)

            real_wav_np = real_wav_resampled.cpu().numpy()
            gen_wav_np = gen_wav_resampled.cpu().numpy()

            stoi_score = stoi(real_wav_np, gen_wav_np, self.target_sr, extended=False)
            total_stoi += stoi_score

        return total_stoi / batch_size

@metrics_registry.add_to_registry(name='si_sdr')
class SISDR:
    def __init__(self, config, target_sr=16000):
        self.target_sr = target_sr
        self.orig_sr = config.mel.sampling_rate
        self.device = config.exp.device
        self.resampler = T.Resample(orig_freq=config.mel.sampling_rate, new_freq=self.target_sr).to(self.device)

    def resample(self, wav):
        return self.resampler(wav.unsqueeze(0)).squeeze(0)

    def si_sdr(self, reference, estimation, eps=1e-8):
        if reference.size() != estimation.size():
            raise ValueError("Signal sizes must be equal")

        reference_energy = torch.sum(reference ** 2) + eps
        scaling_factor = torch.sum(reference * estimation) / reference_energy
        projection = scaling_factor * reference

        noise = estimation - projection

        target_energy = torch.sum(projection ** 2) + eps
        noise_energy = torch.sum(noise ** 2) + eps

        si_sdr_value = 10 * torch.log10(target_energy / noise_energy)
        return si_sdr_value

    def __call__(self, real_batch, gen_batch):
        total_si_sdr = 0.0
        batch_size = len(real_batch['wav'])

        for i in range(batch_size):
            real_wav = real_batch['wav'][i].to(self.device)
            gen_wav = gen_batch['gen_wav'][i]

            real_wav_resampled = self.resample(real_wav)
            gen_wav_resampled = self.resample(gen_wav)

            si_sdr_score = self.si_sdr(real_wav_resampled, gen_wav_resampled)
            total_si_sdr += si_sdr_score

        return total_si_sdr / batch_size    

def extract_prefix(prefix, weights):
    result = OrderedDict()
    for key in weights:
        if key.find(prefix) == 0:
            result[key[len(prefix) :]] = weights[key]
    return result


class Wav2Vec2MOS(nn.Module):
    sample_rate = 16_000

    def __init__(self, path, freeze=True):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.freeze = freeze

        self.dense = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

        if self.freeze:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        state_dict = torch.load(path)["state_dict"]
        self.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
        self.eval()
        self.cuda()

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    def forward(self, x):
        x = self.encoder(x)["last_hidden_state"]
        x = self.dense(x)
        return x.mean(dim=1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()


@metrics_registry.add_to_registry(name="mosnet")
class MOSNet:
    def __init__(self, config):
        self.model = Wav2Vec2MOS("metrics/weights/wave2vec2mos.pth").to(config.exp.device)
        self.resampler = T.Resample(orig_freq=config.mel.sampling_rate, new_freq=self.model.sample_rate).to(config.exp.device)

    def __call__(self, real_batch, gen_batch):
        mos_scores = []
        batch_size = len(gen_batch["gen_wav"])

        for i in range(batch_size):
            gen_wav = gen_batch["gen_wav"][i]

            gen_wav = gen_wav / gen_wav.abs().max()

            gen_wav_resampled = self.resampler(gen_wav.unsqueeze(0)).squeeze(0)

            input_values = self.model.processor(
                gen_wav_resampled.cpu().numpy(), return_tensors="pt", sampling_rate=self.model.sample_rate
            ).input_values.to("cuda")

            with torch.no_grad():
                mos_score = self.model(input_values).item()
                mos_scores.append(mos_score)

        return np.mean(mos_scores)
