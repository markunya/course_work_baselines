import torch
import torchaudio
import random
import os
import numpy as np
import torchaudio.functional as F
from torchaudio.io import AudioEffector, CodecConfig
from scipy import signal
from torch import nn
from utils.data_utils import read_file_list
from utils.class_registry import ClassRegistry
from torchaudio.transforms import Resample

augmentations_registry = ClassRegistry()

def apply_with_prob(forward):
    def wrapper(self, wav):
        if np.random.binomial(n=1,p=self.prob,size=1)[0] == 0:
            return wav
        return forward(self, wav)
    return wrapper

@augmentations_registry.add_to_registry(name='noise')
class RandomNoise(nn.Module):
    def __init__(self, root, noise_files_path, sr, snr_range=(-5, 10), prob=1.0):
        super().__init__()
        self.prob = prob
        self.snr_range = snr_range
        self.root = root
        self.sr = sr
        self.noise_filenames = read_file_list(noise_files_path)

    def _load_noise(self, target_length, target_sr):
        noise_path = os.path.join(self.root, random.choice(self.noise_filenames))
        noise_waveform, noise_sr = torchaudio.load(noise_path)

        if noise_sr != target_sr:
            resampler = Resample(orig_freq=noise_sr, new_freq=target_sr)
            noise_waveform = resampler(noise_waveform)

        noise_waveform = noise_waveform[:, :target_length]
        if noise_waveform.shape[1] < target_length:
            repeat_factor = target_length // noise_waveform.shape[1] + 1
            noise_waveform = noise_waveform.repeat(1, repeat_factor)[:, :target_length]

        return noise_waveform

    @apply_with_prob
    def forward(self, wav):
        target_length = wav.shape[1]
        noise = self._load_noise(target_length, self.sr)
        snr = random.uniform(*self.snr_range)
        noisy_wav = F.add_noise(wav, noise, snr)
        return noisy_wav
    
@augmentations_registry.add_to_registry(name='impulse_response')
class RandomImpulseResponse(torch.nn.Module):
    def __init__(self, root, ir_files_path, sr, prob=1.0):
        super().__init__()
        self.prob = prob
        self.root = root
        self.ir_filenames = read_file_list(ir_files_path)
        self.sr = sr

    def _load_ir(self, target_sr):
        ir_path = os.path.join(self.root, random.choice(self.ir_filenames))
        ir_waveform, sr_ir = torchaudio.load(ir_path)

        if sr_ir != target_sr:
            resampler = Resample(orig_freq=sr_ir, new_freq=target_sr)
            ir_waveform = resampler(ir_waveform)

        ir_waveform = ir_waveform / torch.max(torch.abs(ir_waveform))
        return ir_waveform

    @apply_with_prob
    def forward(self, wav):
        ir_waveform = self._load_ir(self.sr)

        ir_waveform = ir_waveform.unsqueeze(0)
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)

        processed_wav = torch.from_numpy(
            signal.fftconvolve(wav.numpy(), ir_waveform.numpy(), mode='full')
        )

        return processed_wav.squeeze(0)

@augmentations_registry.add_to_registry(name='acrusher')
class RandomAcrusher(nn.Module):
    def __init__(self, sr, bits_range=(1, 9), prob=1.0):
        super().__init__()
        self.prob = prob
        self.sr = sr
        self.bits_range = bits_range

    @apply_with_prob
    def forward(self, wav: torch.Tensor):
        bits = random.randint(*self.bits_range)
        effector = AudioEffector(effect=f"acrusher=bits={bits}")
        return effector.apply(wav.T, self.sr).T.clamp(-1.0,1.0)

    
@augmentations_registry.add_to_registry(name='crystalizer')
class RandomCrystalizer(nn.Module):
    def __init__(self, sr, intensity_range=(1, 4), prob=1.0):
        super().__init__()
        self.prob = prob
        self.sr = sr
        self.intensity_range = intensity_range

    @apply_with_prob
    def forward(self, wav: torch.Tensor):
        intensity = random.randint(*self.intensity_range)
        effector = AudioEffector(effect=f"crystalizer=i={intensity}")
        return effector.apply(wav.T, self.sr).T.clamp(-1.0,1.0)

@augmentations_registry.add_to_registry(name='flanger')
class RandomFlanger(nn.Module):
    def __init__(self, sr, depth_range=(1, 8), prob=1.0):
        super().__init__()
        self.prob = prob
        self.sr = sr
        self.depth_range = depth_range

    @apply_with_prob
    def forward(self, wav: torch.Tensor):
        depth = random.randint(*self.depth_range)
        effector = AudioEffector(effect=f"flanger=depth={depth}")
        return effector.apply(wav.T, self.sr).T.clamp(-1.0,1.0)
    
@augmentations_registry.add_to_registry(name='vibrato')
class RandomVibrato(nn.Module):
    def __init__(self, sr, freq_range=(5, 8), prob=1.0):
        super().__init__()
        self.prob = prob
        self.sr = sr
        self.freq_range = freq_range

    @apply_with_prob
    def forward(self, wav: torch.Tensor):
        freq = random.randint(*self.freq_range)
        effector = AudioEffector(effect=f"vibrato=f={freq}")
        return effector.apply(wav.T, self.sr).T.clamp(-1.0,1.0)

@augmentations_registry.add_to_registry(name='codec')
class RandomCodec(nn.Module):
    def __init__(self, sr, codec_types=("mp3", "ogg"), 
                ogg_encoders=("vorbis", "opus"),
                mp3_bitrate_range=(4000, 16000), prob=1.0):
        super().__init__()
        self.sr = sr
        self.codec_types = codec_types
        self.ogg_encoders = ogg_encoders
        self.mp3_bitrate_range = mp3_bitrate_range
        self.prob = prob

    @apply_with_prob
    def forward(self, wav: torch.Tensor):
        wav = wav.T

        codec = random.choice(self.codec_types)
        codec_config = None
        if codec == 'mp3':
            bitrate = random.randint(*self.mp3_bitrate_range)
            codec_config = CodecConfig(format=codec, encoder="libmp3lame", bitrate=bitrate)
        elif codec == 'ogg':
            encoder = random.choice(self.ogg_encoders)
            codec_config = CodecConfig(format=codec, encoder=encoder)

        effector = AudioEffector(effect=None, codec_config=codec_config)
        processed_wav = effector.apply(wav, self.sr)

        return processed_wav.T
