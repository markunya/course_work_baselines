import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T
from torch import nn
from .loss_builder import losses_registry
from torch_pesq import PesqLoss
from utils.model_utils import unwrap_model, requires_grad
from models.metric_models import UTMOSV2

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
    def __init__(self, target_sr=16000, fft_size=1024, hop_length=256):
        super().__init__()
        self.target_sr = target_sr
        self.fft_size = fft_size
        self.hop_length = hop_length

        self.stft = T.Spectrogram(n_fft=fft_size, hop_length=hop_length, power=1)

    def _extract_features(self, wavlm, input):
        extract_features = unwrap_model(wavlm).feature_extractor(input)
        extract_features = extract_features.transpose(1, 2)
        _, extract_features = unwrap_model(wavlm).feature_projection(extract_features)
        return extract_features

    def forward(self, real_wav, gen_wav, wavlm):
        # it should be guaranteed that wavlm weights not require grad in eval mode
        if len(real_wav.shape) == 3:
            real_wav = real_wav.squeeze(1)
        if len(gen_wav.shape) == 3:
            gen_wav = gen_wav.squeeze(1)

        real_features = self._extract_features(wavlm, real_wav)
        gen_features = self._extract_features(wavlm, gen_wav)

        feature_loss = F.mse_loss(real_features, gen_features)

        real_stft = self.stft(real_wav)
        gen_stft = self.stft(gen_wav)
        stft_loss = F.l1_loss(real_stft, gen_stft)

        lmos_loss = 100 * feature_loss + stft_loss
        return lmos_loss

@losses_registry.add_to_registry(name='pesq')
class PesqLoss_(PesqLoss):
    def __init__(self, factor=0.5, sample_rate=48000):
        super().__init__(factor=factor, sample_rate=sample_rate)

    def forward(self, real_wav, gen_wav):
        return -torch.mean(super().forward(real_wav.squeeze(), gen_wav.squeeze()))

@losses_registry.add_to_registry(name='utmos')
class UTMOSLoss(nn.Module):
    def __init__(
        self,
        sample_rate=48000,
        use_grad_chp=True,
        device=None
    ):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        utmos = UTMOSV2(orig_sr=sample_rate, device=device)

        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            utmos = nn.DataParallel(utmos)

        self.utmos = utmos.to(device)
        self.utmos.eval()
        requires_grad(self.utmos, False)

        if use_grad_chp:
            unwrap_model(self.utmos).utmos.ssl.encoder.model.gradient_checkpointing_enable()
            for backbone in unwrap_model(self.utmos).utmos.spec_long.backbones:
                backbone.set_grad_checkpointing(True)

    def forward(self, gen_wav):
        if gen_wav.ndim == 3:
            gen_wav = gen_wav.squeeze(1)
        
        mos = self.utmos(gen_wav)
            
        return -mos.mean()
