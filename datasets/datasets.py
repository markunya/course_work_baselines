import torch
import random
import numpy as np
import os
import math

from torch.utils.data import Dataset
from librosa.util import normalize

from utils.class_registry import ClassRegistry
from utils.data_utils import load_wav
from utils.data_utils import MAX_WAV_VALUE
from utils.data_utils import mel_spectrogram

datasets_registry = ClassRegistry()

@datasets_registry.add_to_registry(name="meldataset")
class MelDataset(Dataset):
    def __init__(self, files_list, root, config):
        self.files_list = files_list
        self.root = root
        self.fine_tuning = config.fine_tuning
        self.split = config.split
        self.segment_size = config.segment_size
        self.sampling_rate = config.sampling_rate
        self.n_fft = config.n_fft
        self.num_mels = config.num_mels
        self.hop_size = config.hop_size
        self.win_size = config.win_size
        self.fmin = config.fmin
        self.fmax = config.fmax
        self.fmax_loss = config.fmax_for_loss
        
        self.cached_wav = None
        self.n_cache_reuse = 1
        self._cache_ref_count = 0
        self.base_mels_path = None

    def __getitem__(self, index):
        filename = self.files_list[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(os.path.join(self.root, filename))
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio

            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                center=False)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_for_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                center=False)

        return {'mel': mel.squeeze(),
                'wav': audio.squeeze(0),
                'name': filename,
                'mel_for_loss': mel_for_loss.squeeze()}

    def __len__(self):
        return len(self.files_list)
