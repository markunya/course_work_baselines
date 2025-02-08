import torch
import torchaudio
import torchaudio.transforms as T
from torch import nn
from .loss_builder import losses_registry

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
    def __init__(self, target_sr=16000, fft_size=1024, hop_length=256, extraction_layer=7):
        super().__init__()
        self.target_sr = target_sr
        self.fft_size = fft_size
        self.hop_length = hop_length

        self.extraction_layer = extraction_layer
        self.stft = T.Spectrogram(n_fft=fft_size, hop_length=hop_length, power=1)

    def forward(self, real_wav, gen_wav, wavlm):
        # it should be guaranteed that wavlm weights not require grad
        if len(real_wav.shape) == 3:
            real_wav = real_wav.squeeze(1)
        if len(gen_wav.shape) == 3:
            gen_wav = gen_wav.squeeze(1)        
        real_features = wavlm.extract_features(real_wav)[0][self.extraction_layer]
        gen_features = wavlm.extract_features(gen_wav)[0][self.extraction_layer]

        feature_loss = torch.norm(real_features - gen_features, p=2, dim=-1).mean()

        real_stft = self.stft(real_wav)
        gen_stft = self.stft(gen_wav)
        stft_loss = torch.norm(real_stft - gen_stft, p=1, dim=-1).mean()

        lmos_loss = 100 * feature_loss + stft_loss
        return lmos_loss
