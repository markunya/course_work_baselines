import torch
import random
import numpy as np
import os
import librosa
import math
import torchaudio

from torch.utils.data import Dataset
from librosa.util import normalize

from typing import Literal
from utils.class_registry import ClassRegistry
from utils.data_utils import load_wav, MAX_WAV_VALUE, mel_spectrogram, read_file_list, split_audios
from utils.data_utils import low_pass_filter
from utils.model_utils import closest_power_of_two
from datasets.augmentations import augmentations_registry
from tqdm import tqdm

datasets_registry = ClassRegistry()

WAV_AFTERNORM_COEF = 0.95

@datasets_registry.add_to_registry(name='mel_dataset')
class MelDataset(Dataset):
    def __init__(self, root, files_list_path, mel_conf, fine_tuning=False, split=True, eval=False):
        self.root = root
        self.files_list = read_file_list(files_list_path)
        self.fine_tuning = fine_tuning
        self.split = split
        self.segment_size = mel_conf.segment_size
        self.sampling_rate = mel_conf.in_sr
        self.n_fft = mel_conf.n_fft
        self.num_mels = mel_conf.num_mels
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
            tqdm.write(f'Warning: Failed to load {filename}, error: {e}. Picking a random file instead.')
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
        self.sampling_rate = mel_conf.in_sr
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
class VoicebankDataset(Dataset):
    def __init__(
        self,
        root,
        files_list_path,
        mel_conf,
        noisy_wavs_dir,
        clean_wavs_dir,
        split=True,
        eval=False,
        input_freq=None,
    ):
        if clean_wavs_dir:
            clean_wavs_dir = os.path.join(root, clean_wavs_dir)
        noisy_wavs_dir = os.path.join(root, noisy_wavs_dir)
        self.files_list = read_file_list(files_list_path)

        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        self.segment_size = mel_conf.segment_size
        self.sampling_rate = mel_conf.in_sr
        self.split = split
        self.input_freq = input_freq

    def __getitem__(self, index):
        fn = self.files_list[index] + '.wav'

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
        pad_size = closest_power_of_two(input_wav.shape[-1]) - input_wav.shape[-1]
        input_wav = torch.nn.functional.pad(input_wav, (0, pad_size))
        pad_size = closest_power_of_two(wav.shape[-1]) - wav.shape[-1]
        wav = torch.nn.functional.pad(wav, (0, pad_size))

        return {
                'input_wav': input_wav.squeeze(),
                'wav': wav.squeeze(),
                'name': fn
            }

    def __len__(self):
        return len(self.files_list)

class AugmentedDataset(Dataset):
    def __init__(
        self,
        root,
        files_list_path,
        mel_conf,
        seed=42,
        eval=False,
        split=True,
        augs_conf=tuple()
    ):
        self.root = root
        self.files_list = read_file_list(files_list_path)
        self.segment_size = mel_conf.segment_size
        self.in_sr = mel_conf.in_sr
        self.out_sr = mel_conf.out_sr if 'out_sr' in mel_conf else self.in_sr
        self.split = split
        self.eval = eval
        self.seed = seed
        self.augmentations = self._get_augs_from_conf(augs_conf)
        self.resamplers = {}

    def _get_augs_from_conf(self, augs_conf):
        augs = []
        if not augs_conf:
            return augs

        for aug in augs_conf:
            name = aug['name']
            args = aug['args'].copy()

            if name == 'noise':
                args['noise_files_path'] = args['noise_files_path']['val'] \
                                            if self.eval else args['noise_files_path']['train']
            elif name == 'impulse_response':
                args['ir_files_path'] = args['ir_files_path']['val'] \
                                            if self.eval else args['ir_files_path']['train']

            try:
                augs.append(augmentations_registry[name](sr=self.out_sr, **args))
            except Exception as e:
                tqdm.write(f"Failed to initiallize {name}: {e}")

        return augs

    def _apply_augs(self, wav, index):
        result = wav.clone()
        seed = self.seed + index if self.eval else None

        for aug in self.augmentations:
            try:
                result = aug(result, seed)
            except Exception as e:
                tqdm.write(f"Failed to apply {aug.__class__.__name__}: {e}")
        
        return result

    def _safe_normalize(self, wav):
        wav = torch.nan_to_num(wav)
        max_val = torch.max(torch.abs(wav))
        return wav / max_val if max_val > 0 else wav

    def __len__(self):
        return len(self.files_list)

@datasets_registry.add_to_registry(name='augmented_libritts-r')
class AugmentedLibriTTSR(AugmentedDataset):
    def __init__(
        self,
        root,
        files_list_path,
        mel_conf,
        seed=42,
        eval=False,
        split=True,
        augs_conf=tuple()
    ):
        super().__init__(
            root=root,
            files_list_path=files_list_path,
            mel_conf=mel_conf,
            seed=seed,
            eval=eval,
            split=split,
            augs_conf=augs_conf
        )

    
    def __getitem__(self, index):
        filename = self.files_list[index]

        wav, sr = torchaudio.load(os.path.join(self.root, filename))            

        if self.in_sr != sr:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(
                                        orig_freq=sr,
                                        new_freq=self.in_sr
                                    )
            wav = self.resamplers[sr](wav.squeeze(0)).numpy()
        else:
            wav = wav.numpy().squeeze(0)

        (wav,) = split_audios([wav], self.segment_size, self.split)

        augmented = self._apply_augs(torch.from_numpy(wav[None]), index).squeeze()

        input_wav = self._safe_normalize(augmented)[None] * WAV_AFTERNORM_COEF
        target_wav = self._safe_normalize(torch.from_numpy(wav)) * WAV_AFTERNORM_COEF

        input_wav = input_wav[:,:target_wav.shape[-1]]
        pad_size = closest_power_of_two(target_wav.shape[-1]) - target_wav.shape[-1]

        input_wav = torch.nn.functional.pad(input_wav, (0, pad_size)).squeeze()
        target_wav = torch.nn.functional.pad(target_wav, (0, pad_size)).squeeze()

        assert input_wav.shape == target_wav.shape
        return {
            'input_wav': input_wav,
            'wav': target_wav,
            'name': filename
        }

@datasets_registry.add_to_registry(name='augmented_daps')
class AugmentedDaps(AugmentedDataset):
    def __init__(
        self,
        root,
        files_list_path,
        mel_conf,
        seed=42,
        eval=False,
        split=True,
        virtual_len=100000,
        augs_conf=tuple()
    ):
        super().__init__(
            root=root,
            files_list_path=files_list_path,
            mel_conf=mel_conf,
            seed=seed,
            eval=eval,
            split=split,
            augs_conf=augs_conf
        )
        assert self.out_sr % self.in_sr == 0, "in_sr should devide out_sr"
        if not eval and not split:
            raise ValueError('This dataset do not support split=False in train mode')
        
        self.virtual_len = virtual_len
        self.cache = {}
    
    def __getitem__(self, index):
        if not self.eval:
            index = index % len(self.files_list)

        filename = self.files_list[index]
        if filename not in self.cache: 
            self.cache[filename] = torchaudio.load(os.path.join(self.root, filename))         
        wav, sr = self.cache[filename]

        if self.out_sr != sr:
            key = (sr, self.out_sr)
            if key not in self.resamplers:
                self.resamplers[key] = torchaudio.transforms.Resample(
                                        orig_freq=sr,
                                        new_freq=self.out_sr
                                    )
            wav = self.resamplers[key](wav.squeeze(0)).numpy()
        else:
            wav = wav.numpy().squeeze(0)

        (wav,) = split_audios([wav], self.segment_size, self.split)
        wav = torch.from_numpy(wav[None])

        input_wav = self._apply_augs(wav, index).squeeze()[None]
        
        target_wav = self._safe_normalize(wav) * WAV_AFTERNORM_COEF

        input_wav = input_wav[:,:target_wav.shape[-1]]

        base = 3072
        remainder = input_wav.shape[-1] % base
        pad_size = (base - remainder) if remainder != 0 else 0
        input_wav = torch.nn.functional.pad(input_wav, (0, pad_size))
        target_wav = torch.nn.functional.pad(target_wav, (0, pad_size))

        if self.in_sr != self.out_sr:
            key = (self.out_sr, self.in_sr)
            if key not in self.resamplers:
                self.resamplers[key] = torchaudio.transforms.Resample(
                                        orig_freq=self.out_sr,
                                        new_freq=self.in_sr
                                    )
            input_wav = self.resamplers[key](input_wav)

        input_wav = self._safe_normalize(input_wav) * WAV_AFTERNORM_COEF

        return {
            'input_wav': input_wav.squeeze(),
            'wav': target_wav.squeeze(),
            'name': filename
        }
    
    def __len__(self):
        if self.eval:
            return len(self.files_list)
        return self.virtual_len