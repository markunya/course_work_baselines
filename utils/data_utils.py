import os
import torch
import torchaudio
import torch.utils.data
import numpy as np
import omegaconf
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from omegaconf import OmegaConf

MAX_WAV_VALUE = 32768.0

def apply_inheritance(global_config):
    def apply_inheritance_impl(config):
        nonlocal global_config
        if isinstance(config, omegaconf.dictconfig.DictConfig):
            if 'inherit' in config:
                parent_config_name = config.inherit
                if parent_config_name in global_config:
                    parent_config = global_config[parent_config_name]
                    config = OmegaConf.merge(parent_config, config)
                    del config['inherit']
            
            for key, value in config.items():
                config[key] = apply_inheritance_impl(value)
        return config
    return apply_inheritance_impl(global_config)

def load_config():
    conf_cli = OmegaConf.from_cli()
    config_path = conf_cli.exp.config_path
    conf_file = OmegaConf.load(config_path)
    config = OmegaConf.merge(conf_file, conf_cli)
    config = apply_inheritance(config)
    return config

def save_wavs_to_dir(wavs_dict, path, sample_rate):
    os.makedirs(path, exist_ok=True)
    for name, wav in wavs_dict.items():
        file_path = os.path.join(path, f"{name}.wav")
        torchaudio.save(file_path, wav.unsqueeze(0), sample_rate)

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                    center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def read_file_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        file_list = [line.strip() for line in f.readlines() if line.strip()]
    return file_list
