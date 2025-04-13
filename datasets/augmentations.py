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

def apply_with_prob(forward, seed=None):
    def wrapper(self, wav, seed=seed):
        rnd = np.random.RandomState(seed)
        if rnd.binomial(n=1,p=self.prob,size=1)[0] == 0:
            return wav
        return forward(self, wav, seed)
    return wrapper

def get_rnd(seed=None):
    if seed is None:
        return random.Random()
    return random.Random(seed)

@augmentations_registry.add_to_registry(name='noise')
class RandomNoise:
    def __init__(self, root, noise_files_path, sr, snr_range=(10, 20), prob=1.0):
        super().__init__()
        self.prob = prob
        self.snr_range = snr_range
        self.root = root
        self.sr = sr
        self.noise_filenames = read_file_list(noise_files_path)
        self.resampler_cache = {}

    def _load_noise(self, target_length, target_sr, rnd):
        noise_path = os.path.join(self.root, rnd.choice(self.noise_filenames))
        noise_waveform, noise_sr = torchaudio.load(noise_path)

        if noise_sr != target_sr:
            if noise_sr not in self.resampler_cache:
                self.resampler_cache[noise_sr] = Resample(noise_sr, target_sr)
            noise_waveform = self.resampler_cache[noise_sr](noise_waveform)

        repeat_factor = (target_length // noise_waveform.shape[1]) + 1
        noise_waveform = noise_waveform.repeat(1, repeat_factor)
        noise_waveform = noise_waveform[:, :target_length]

        return noise_waveform

    @apply_with_prob
    def __call__(self, wav, seed=None):
        rnd = get_rnd(seed)
        target_length = wav.shape[1]
        noise = self._load_noise(target_length, self.sr, rnd)
        snr = torch.tensor([rnd.uniform(*self.snr_range)])
        noisy_wav = F.add_noise(wav, noise, snr)
        return noisy_wav
    
@augmentations_registry.add_to_registry(name='impulse_response')
class RandomImpulseResponse:
    def __init__(self, root, ir_files_path, sr, prob=1.0):
        super().__init__()
        self.prob = prob
        self.root = root
        self.ir_filenames = read_file_list(ir_files_path)
        self.sr = sr
        self.resampler_cache = {}

    def _load_ir(self, target_sr, rnd):
        ir_path = os.path.join(self.root, rnd.choice(self.ir_filenames))
        ir_waveform, sr_ir = torchaudio.load(ir_path)

        if sr_ir != target_sr:
            if sr_ir not in self.resampler_cache:
                self.resampler_cache[sr_ir] = Resample(sr_ir, target_sr)
            ir_waveform = self.resampler_cache[sr_ir](ir_waveform)

        ir_waveform = ir_waveform / torch.max(torch.abs(ir_waveform))
        return ir_waveform

    @apply_with_prob
    def __call__(self, wav, seed=None):
        rnd = get_rnd(seed)

        ir_waveform = self._load_ir(self.sr, rnd)
        ir_waveform = ir_waveform.unsqueeze(0)
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)

        processed_wav = F.fftconvolve(wav, ir_waveform, mode='full')
        
        return processed_wav.squeeze(0)

@augmentations_registry.add_to_registry(name='acrusher')
class RandomAcrusher:
    def __init__(self, sr, bits_range=(1, 9), prob=1.0):
        super().__init__()
        self.prob = prob
        self.sr = sr
        self.bits_range = bits_range
        self.effector_cache = {}

    @apply_with_prob
    def __call__(self, wav, seed=None):
        rnd = get_rnd(seed)
        bits = rnd.randint(*self.bits_range)
        if bits not in self.effector_cache:
            self.effector_cache[bits] = AudioEffector(effect=f"acrusher=bits={bits}", pad_end=False)
        return self.effector_cache[bits].apply(wav.T, self.sr).T.clamp(-1.0,1.0)
    
@augmentations_registry.add_to_registry(name='crystalizer')
class RandomCrystalizer:
    def __init__(self, sr, intensity_range=(1, 4), prob=1.0):
        super().__init__()
        self.prob = prob
        self.sr = sr
        self.intensity_range = intensity_range
        self.effector_cache = {}

    @apply_with_prob
    def __call__(self, wav, seed=None):
        rnd = get_rnd(seed)
        intensity = rnd.randint(*self.intensity_range)
        if intensity not in self.effector_cache:
            self.effector_cache[intensity] = AudioEffector(effect=f"crystalizer=i={intensity}", pad_end=False)
        return self.effector_cache[intensity].apply(wav.T, self.sr).T.clamp(-1.0,1.0)

@augmentations_registry.add_to_registry(name='flanger')
class RandomFlanger:
    def __init__(self, sr, depth_range=(1, 8), prob=1.0):
        super().__init__()
        self.prob = prob
        self.sr = sr
        self.depth_range = depth_range
        self.effector_cache = {}

    @apply_with_prob
    def __call__(self, wav, seed=None):
        rnd = get_rnd(seed)
        depth = rnd.randint(*self.depth_range)
        if depth not in self.effector_cache:
            self.effector_cache[depth] = AudioEffector(effect=f"flanger=depth={depth}", pad_end=False)
        return self.effector_cache[depth].apply(wav.T, self.sr).T.clamp(-1.0,1.0)
    
@augmentations_registry.add_to_registry(name='vibrato')
class RandomVibrato:
    def __init__(self, sr, freq_range=(5, 8), prob=1.0):
        super().__init__()
        self.prob = prob
        self.sr = sr
        self.freq_range = freq_range
        self.effector_cache = {}

    @apply_with_prob
    def __call__(self, wav, seed=None):
        rnd = get_rnd(seed)
        freq = rnd.randint(*self.freq_range)

        if freq not in self.effector_cache:
            self.effector_cache[freq] = AudioEffector(effect=f"vibrato=f={freq}", pad_end=False)

        out = self.effector_cache[freq].apply(wav.T, self.sr).T.clamp(-1.0,1.0)
        out = out.clamp(-1.0, 1.0)
        out[torch.isnan(out)] = 0.0
        out[torch.isinf(out)] = 0.0
        return out

@augmentations_registry.add_to_registry(name='codec')
class RandomCodec:
    def __init__(self, sr, codec_types=("mp3", "ogg"), 
                ogg_encoders=("libvorbis", "libopus"),
                mp3_bitrate_range=(4000, 16000), prob=1.0):
        super().__init__()
        self.sr = sr
        self.codec_types = codec_types
        self.ogg_encoders = ogg_encoders
        self.mp3_bitrate_range = mp3_bitrate_range
        self.prob = prob
        self.effector_cache = {}

    @apply_with_prob
    def __call__(self, wav, seed=None):
        rnd = get_rnd(seed)
        wav = wav.T
        codec = rnd.choice(self.codec_types)
        if codec == 'mp3':
            bit_rate = rnd.randint(*self.mp3_bitrate_range)
            if (codec, bit_rate) not in self.effector_cache:
                self.effector_cache[(codec, bit_rate)] = AudioEffector(
                    format=codec,
                    codec_config=CodecConfig(bit_rate=bit_rate)
                )
            effector = self.effector_cache[(codec, bit_rate)]
        elif codec == 'ogg':
            encoder = rnd.choice(self.ogg_encoders)
            if (codec, encoder) not in self.effector_cache:
                self.effector_cache[(codec, encoder)] = AudioEffector(
                    format=codec,
                    encoder=encoder
                )
            effector = self.effector_cache[(codec, encoder)]
        else:
            raise ValueError(f'Invalid codec: {codec}')

        processed_wav = effector.apply(wav, self.sr)
        return processed_wav.T
