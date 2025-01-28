import torch
import random
import numpy as np
import os
import librosa
import math

from torch.utils.data import Dataset
from librosa.util import normalize

from utils.class_registry import ClassRegistry
from utils.data_utils import load_wav, MAX_WAV_VALUE, mel_spectrogram, read_file_list, split_audios,debug_msg
from utils.data_utils import low_pass_filter
from utils.model_utils import closest_power_of_two

datasets_registry = ClassRegistry()

WAV_AFTERNORM_COEF = 0.95

@datasets_registry.add_to_registry(name='mel_dataset')
class MelDataset(Dataset):
    def __init__(self, root, files_list_path, mel_conf, fine_tuning=False, split=True):
        self.root = root
        self.files_list = read_file_list(files_list_path)
        self.fine_tuning = fine_tuning
        self.split = split
        self.segment_size = mel_conf.segment_size
        self.sampling_rate = mel_conf.sampling_rate
        self.n_fft = mel_conf.n_fft
        self.num_mels = mel_conf.mel_confnum_mels
        self.hop_size = mel_conf.hop_size
        self.win_size = mel_conf.win_size
        self.fmin = mel_conf.fmin
        self.fmax = mel_conf.fmax
        self.fmax_loss = mel_conf.fmax_for_loss
        
        self.cached_wav = None
        self.n_cache_reuse = 1
        self._cache_ref_count = 0
        self.base_mels_path = None

    def __getitem__(self, index, attempts=5):
        if attempts == 0:
            raise ValueError('Unable to load a valid file after several attempts.')
        filename = self.files_list[index]
        try:
            if self._cache_ref_count == 0:
                audio, sampling_rate = load_wav(os.path.join(self.root, filename))
                audio = audio / MAX_WAV_VALUE
                if not self.fine_tuning:
                    audio = normalize(audio) * WAV_AFTERNORM_COEF
                self.cached_wav = audio

                if sampling_rate != self.sampling_rate:
                    raise ValueError('{} SR doesn\'t match target {} SR'.format(
                        sampling_rate, self.sampling_rate))
                self._cache_ref_count = self.n_cache_reuse
            else:
                audio = self.cached_wav
                self._cache_ref_count -= 1
        except Exception as e:
            print(f'Warning: Failed to load {filename}, error: {e}. Picking a random file instead.')
            random_index = random.randint(0, len(self.files_list) - 1)
            return self.__getitem__(random_index, attempts=attempts-1)

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

        return {
                'mel': mel.squeeze(),
                'wav': audio.squeeze(0),
                'name': filename,
                'mel_for_loss': mel_for_loss.squeeze()
            }

    def __len__(self):
        return len(self.files_list)

@datasets_registry.add_to_registry(name='vctk')
class VCTKDataset(Dataset):
    def __init__(
        self,
        root,
        files_list_path,
        mel_conf,
        split=True,
        input_freq=None,
        lowpass='default',
    ):
        self.root = root
        self.files_list = read_file_list(files_list_path)
        self.segment_size = mel_conf.segment_size
        self.sampling_rate = mel_conf.sampling_rate
        self.split = split
        self.input_freq = input_freq
        self.lowpass = lowpass

    def __getitem__(self, index):
        vctk_fn = self.files_list[index]

        vctk_wav = librosa.load(
            os.path.join(self.root, vctk_fn),
            sr=self.sampling_rate,
            res_type='polyphase',
        )[0]
        (vctk_wav, ) = split_audios([vctk_wav], self.segment_size, self.split)

        lp_inp = low_pass_filter(
            vctk_wav, self.input_freq,
            lp_type=self.lowpass, orig_sr=self.sampling_rate
        )
        input_wav = normalize(lp_inp)[None] * WAV_AFTERNORM_COEF
        assert input_wav.shape[1] == vctk_wav.size

        input_wav = torch.FloatTensor(input_wav)
        pad_size = closest_power_of_two(input_wav.shape[-1]) - input_wav.shape[-1]
        input_wav = torch.nn.functional.pad(input_wav, (0, pad_size))
        wav = torch.FloatTensor(normalize(vctk_wav) * WAV_AFTERNORM_COEF)
        pad_size = closest_power_of_two(wav.shape[-1]) - wav.shape[-1]
        wav = torch.nn.functional.pad(wav, (0, pad_size))

        return {
                'input_wav': input_wav.squeeze(),
                'wav': wav.squeeze(),
                'name': vctk_fn
            }

    def __len__(self):
        return len(self.files_list)

@datasets_registry.add_to_registry(name='voicebank')
class VoicebankDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        files_list_path,
        mel_conf,
        noisy_wavs_dir,
        clean_wavs_dir=None,
        split=True,
        input_freq=None,
    ):
        if clean_wavs_dir:
            clean_wavs_dir = os.path.join(root, clean_wavs_dir)
        noisy_wavs_dir = os.path.join(root, noisy_wavs_dir)
        self.files_list = read_file_list(files_list_path)

        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        self.segment_size = mel_conf.segment_size
        self.sampling_rate = mel_conf.sampling_rate
        self.split = split
        self.input_freq = input_freq

    def __getitem__(self, index):
        fn = self.files_list[index]

        clean_wav = librosa.load(
            os.path.join(self.clean_wavs_dir, fn),
            sr=self.sampling_rate,
            res_type='polyphase',
        )[0]
        noisy_wav = librosa.load(
            os.path.join(self.noisy_wavs_dir, fn),
            sr=self.sampling_rate,
            res_type='polyphase',
        )[0]
        clean_wav, noisy_wav = split_audios(
            [clean_wav, noisy_wav],
            self.segment_size, self.split
        )

        input_wav = normalize(noisy_wav)[None] * WAV_AFTERNORM_COEF
        assert input_wav.shape[1] == clean_wav.size

        input_wav = torch.FloatTensor(input_wav)
        wav = torch.FloatTensor(normalize(clean_wav) * WAV_AFTERNORM_COEF)

        return {
                'input_wav': input_wav.squeeze(),
                'wav': wav.squeeze(),
                'name': fn
            }

    def __len__(self):
        return len(self.files_list)
