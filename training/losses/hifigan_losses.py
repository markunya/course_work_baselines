import torch
import torch.nn as nn

from .losses import losses_registry

@losses_registry.add_to_registry(name='feature_loss')
class FeatureLoss(nn.Module):
    def forward(self, fmaps_real, fmaps_gen):
        loss = 0
        for one_disc_fmaps_real, one_disc_fmaps_gen in zip(fmaps_real, fmaps_gen):
            for fmap_real, fmap_gen in zip(one_disc_fmaps_real, one_disc_fmaps_gen):
                loss += torch.mean(torch.abs(fmap_real - fmap_gen))
        return loss

@losses_registry.add_to_registry(name='hifigan_disc_loss')
class DiscriminatorLoss(nn.Module):        
    def forward(self, discs_real_out, discs_gen_out):
        loss = 0
        for disc_real_out, disc_gen_out in zip(discs_real_out, discs_gen_out):
            real_one_disc_loss = torch.mean((1 - disc_real_out)**2)
            gen_one_disc_loss = torch.mean(disc_gen_out**2)
            loss += (real_one_disc_loss + gen_one_disc_loss)
        return loss

@losses_registry.add_to_registry(name='hifigan_gen_loss')
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
    