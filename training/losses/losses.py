import torch
import torchaudio
import torchaudio.transforms as T
from torch import nn
from .loss_builder import losses_registry
from torch_pesq import PesqLoss
from utils.model_utils import unwrap_model

@losses_registry.add_to_registry(name='feature_loss')
class FeatureLoss(nn.Module):
    def forward(self, fmaps_real, fmaps_gen):
        loss = 0
        for one_disc_fmaps_real, one_disc_fmaps_gen in zip(fmaps_real, fmaps_gen):
            for fmap_real, fmap_gen in zip(one_disc_fmaps_real, one_disc_fmaps_gen):
                loss += torch.mean(torch.abs(fmap_real - fmap_gen))
        return loss

@losses_registry.add_to_registry(name='disc_loss')
class DiscriminatorLoss(nn.Module):        
    def forward(self, discs_real_out, discs_gen_out):
        loss = 0
        for disc_real_out, disc_gen_out in zip(discs_real_out, discs_gen_out):
            real_one_disc_loss = torch.mean((1 - disc_real_out)**2)
            gen_one_disc_loss = torch.mean(disc_gen_out**2)
            loss += (real_one_disc_loss + gen_one_disc_loss)
        return loss

@losses_registry.add_to_registry(name='gen_loss')
class GeneratorLoss(nn.Module):
    def forward(self, discs_gen_out):
        loss = 0
        for disc_gen_out in discs_gen_out:
            one_disc_loss = torch.mean((1 - disc_gen_out)**2)
            loss += one_disc_loss
        return loss

@losses_registry.add_to_registry(name='l1_mel_loss')
class L1Loss(nn.L1Loss):
    def forward(self, gen_mel, real_mel):
        return super().forward(gen_mel, real_mel)

@losses_registry.add_to_registry(name='lmos')
class LMOSLoss(nn.Module):
    def __init__(self, target_sr=16000, fft_size=1024, hop_length=256, extraction_layer=6):
        super().__init__()
        self.target_sr = target_sr
        self.fft_size = fft_size
        self.hop_length = hop_length

        self.extraction_layer = extraction_layer
        self.stft = T.Spectrogram(n_fft=fft_size, hop_length=hop_length, power=1)

    def forward(self, real_wav, gen_wav, wavlm):
        # it should be guaranteed that wavlm weights not require grad in eval mode
        if len(real_wav.shape) == 3:
            real_wav = real_wav.squeeze(1)
        if len(gen_wav.shape) == 3:
            gen_wav = gen_wav.squeeze(1)        
        real_features = unwrap_model(wavlm).extract_features(real_wav)[0][self.extraction_layer]
        gen_features = unwrap_model(wavlm).extract_features(gen_wav)[0][self.extraction_layer]

        feature_loss = torch.norm(real_features - gen_features, p=2, dim=-1).mean()

        real_stft = self.stft(real_wav)
        gen_stft = self.stft(gen_wav)
        stft_loss = torch.norm(real_stft - gen_stft, p=1, dim=-1).mean()

        lmos_loss = 100 * feature_loss + stft_loss
        return lmos_loss

@losses_registry.add_to_registry(name='pesq')
class PesqLoss_(PesqLoss):
    def __init__(self, factor=0.5, sample_rate=48000):
        super().__init__(factor=factor, sample_rate=sample_rate)

    def forward(self, real_wav, gen_wav):
        super().forward(real_wav, gen_wav)

@losses_registry.add_to_registry(name='utmos')
class UTMOSLoss(nn.Module):
    def __init__(
            self,
            sample_rate=48000,
            ckpt_path="utmos_demo_repo/epoch=3-step=7459.ckpt",
            device=None
        ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = None
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000,
            resampling_method="sinc_interpolation",
            lowpass_filter_width=6,
            dtype=torch.float32,
        ).to(device)
        
    def forward(self, gen_wav):
        if len(gen_wav.shape) == 1:
            out_wavs = gen_wav.unsqueeze(0).unsqueeze(0)
        elif len(gen_wav.shape) == 2:
            out_wavs = gen_wav.unsqueeze(0)
        elif len(gen_wav.shape) == 3:
            out_wavs = gen_wav
        else:
            raise ValueError('Dimension of input tensor needs to be <= 3.')
        
        if self.in_sr != 16000:
                out_wavs = self.resampler(out_wavs)
        bs = out_wavs.shape[0]
        batch = {
            'wav': out_wavs,
            'domains': torch.zeros(bs, dtype=torch.int).to(self.device),
            'judge_id': torch.ones(bs, dtype=torch.int).to(self.device)*288
        }
        with torch.no_grad():
            output = self.model(batch)
            
        return torch.mean(output.mean(dim=1).squeeze(1)*2 + 3)
