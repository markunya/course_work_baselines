import utmosv2
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio.transforms as T
from torch import nn
import torchvision
import utmosv2
from transformers import AutoModel
import torch
import torch.nn as nn
import wvmos
import os
from utils.model_utils import requires_grad

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

class SSLEncoder(nn.Module):
    def __init__(self, sr: int, model_name: str, freeze: bool):
        super().__init__()
        self.sr = sr
        self.model = AutoModel.from_pretrained(model_name)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        if x.dim() == 3:
            x = x.squeeze(1)

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        x = (x - mean) / std

        x = x.to(self.model.device)
        outputs = self.model(input_values=x, output_hidden_states=True)

        return outputs.hidden_states

class UTMOSV2(nn.Module):
    def __init__(self, orig_sr):
        super().__init__()
        self.orig_sr = orig_sr
        self.target_sr = 16000
        self.ssl_duration = 3
        self.spec_frames_frame_sec = 1.4
        self.spec_frames_num_frames = 2

        spec_cfg_defaults = dict(
            n_fft=4096,
            hop_length=32,
            n_mels=512,
            shape=(512, 512),
            norm=80,
        )
        self.spec_cfgs = [
            dict(**spec_cfg_defaults, win_length=4096),
            dict(**spec_cfg_defaults, win_length=2048),
            dict(**spec_cfg_defaults, win_length=1024),
            dict(**spec_cfg_defaults, win_length=512),
        ]
        self.valid_transform = torchvision.transforms.Resize((512, 512))
        self.spec_frames_mixup_inner = True
        self.spec_frames_mixup_alpha = 0.4
        
        self.utmos = utmosv2.create_model()
        self.utmos._model.ssl.encoder = SSLEncoder(
            sr=self.target_sr,
            model_name="facebook/wav2vec2-base",
            freeze=True
        )

        self.resample = T.Resample(orig_freq=self.orig_sr, new_freq=self.target_sr)
        self.mel_spectrograms = [
            T.MelSpectrogram(
                sample_rate=self.target_sr,
                n_fft=cfg["n_fft"],
                hop_length=cfg["hop_length"],
                win_length=cfg["win_length"],
                n_mels=cfg["n_mels"],
                power=2.0,
            ) for cfg in self.spec_cfgs
        ]

        requires_grad(self, False)

    def _extend_audio(self, y: torch.Tensor, length: int, mode: str = "tile") -> torch.Tensor:
        if y.shape[-1] >= length:
            return y[:, :length]
        elif mode == "tile":
            n = length // y.shape[-1] + 1
            y = y.repeat(1, n)[:, :length]
            return y
        else:
            raise NotImplementedError

    def _select_random_start(self, y: torch.Tensor, length: int) -> torch.Tensor:
        start = torch.randint(0, y.shape[-1] - length + 1, (1,)).item()
        return y[:, start : start + length]

    def _make_melspec(self, y: torch.Tensor, mel_transform: T.MelSpectrogram, norm: float) -> torch.Tensor:
        spec = mel_transform(y)

        ref_value = spec.amax(dim=[1, 2], keepdim=True)
        spec_db = 10 * (torch.log10(spec + 1e-10) - torch.log10(ref_value + 1e-10))

        if norm is not None:
            spec_db = (spec_db + norm) / norm

        spec_db = spec_db.unsqueeze(1).repeat(1, 3, 1, 1)

        return spec_db


    def _getitem_ssl(self, y: torch.Tensor) -> torch.Tensor:
        length = int(self.ssl_duration * self.target_sr)
        y = self._extend_audio(y, length, mode="tile")
        y = self._select_random_start(y, length)
        return y

    def _getitem_multispec(self, y: torch.Tensor) -> torch.Tensor:
        B = y.shape[0]
        specs = []
        length = int(self.spec_frames_frame_sec * self.target_sr)
        y = self._extend_audio(y, length)

        for _ in range(self.spec_frames_num_frames):
            y1 = self._select_random_start(y, length)
            batch_specs = []
            
            for mel_transform, spec_cfg in zip(self.mel_spectrograms, self.spec_cfgs):
                spec = self._make_melspec(y1, mel_transform, spec_cfg["norm"])

                if self.spec_frames_mixup_inner:
                    y2 = self._select_random_start(y, length)
                    spec2 = self._make_melspec(y2, mel_transform, spec_cfg["norm"])

                    lmd = torch.distributions.Beta(
                        self.spec_frames_mixup_alpha,
                        self.spec_frames_mixup_alpha
                    ).sample()
                    spec = lmd * spec + (1 - lmd) * spec2

                spec = self.valid_transform(spec)
                batch_specs.append(spec)

            specs.append(torch.stack(batch_specs, dim=1))

        return torch.cat(specs, dim=1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.orig_sr != self.target_sr:
            y = self.resample(y)

        ssl_input = self._getitem_ssl(y)
        spec_input = self._getitem_multispec(y)

        d = torch.zeros((y.shape[0], 10))
        d[:, 1] = 1

        return self.utmos(ssl_input, spec_input, d)

class WV_MOS(nn.Module):
    def __init__(self, cuda=True):
        super().__init__()
        self.encoder = wvmos.wv_mos.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        path = os.path.join(os.path.expanduser('~'), ".cache/wv_mos/wv_mos.ckpt")
        
        self.dense = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        
        checkpoint = torch.load(
            path,
            weights_only=False,
            map_location='cpu' if not cuda else 'cuda'
        )['state_dict']
        self.load_state_dict(
            wvmos.wv_mos.extract_prefix('model.', checkpoint)
        )
        
        self.eval()
        if cuda:
            self.cuda()
        
    def forward(self, x):
        x = self.encoder(x)['last_hidden_state']
        x = self.dense(x)
        x = x.mean(dim=[1,2], keepdims=True)
        return x
