import torch
import torch.nn as nn
import inspect
from utils.class_registry import ClassRegistry

losses_registry = ClassRegistry()

class LossBuilder:
    def __init__(self, config):
        self.losses = {}
        self.coefs = {}

        for loss_name, loss_coef in config['losses'].items():
            self.coefs[loss_name] = loss_coef
            loss_args = {}
            if 'losses_args' in config and loss_name in config['losses_args']:
                loss_args = config['losses_args']['loss_name']
            self.losses[loss_name] = losses_registry[loss_name](**loss_args)

    def calculate_loss(self, info):
        loss_dict = {}
        loss_dict['total_loss'] = 0.0

        for loss_name, loss in self.losses.items():
            signature = inspect.signature(loss.forward)
            param_names = [param.name for param in signature.parameters.values()]

            current_loss = 0.0
            for suffix, kwargs in info.items():
                if suffix != '':
                    suffix = '_' + suffix

                loss_args = {param: kwargs[param] for param in param_names if param in kwargs}
                loss_val = loss(**loss_args)
                current_loss += loss_val
                loss_dict[loss_name + suffix] = float(loss_val)
    
            loss_dict['total_loss'] += self.coefs[loss_name] * current_loss

        return loss_dict

@losses_registry.add_to_registry(name='feature_loss')
class FeatureLoss(nn.Module):
    def forward(self, fmaps_real, fmaps_gen):
        dim = tuple(range(2, fmaps_gen.dim()))
        l1_mean = torch.mean(torch.abs(fmaps_real - fmaps_gen), dim=dim)
        return torch.sum(l1_mean)

@losses_registry.add_to_registry(name='hifigan_disc_loss')
class DiscriminatorLoss(nn.Module):        
    def forward(self, discs_real_out, discs_gen_out):
        dim = tuple(range(1, discs_real_out.dim()))
        real_losses = torch.mean((1 - discs_real_out) ** 2, dim=dim)
        gen_losses = torch.mean(discs_gen_out ** 2, dim=1)
        loss = torch.sum(real_losses + gen_losses)
        return loss

@losses_registry.add_to_registry(name='hifigan_gen_loss')
class GeneratorLoss(nn.Module):
    def forward(self, discs_gen_out):
        dim = tuple(range(1, discs_gen_out.dim()))
        gen_losses = torch.mean((1 - discs_gen_out)**2, dim=dim)
        loss = torch.sum(gen_losses)
        return loss

@losses_registry.add_to_registry(name='l1_mel_loss')
class L1Loss(nn.L1Loss):
    def forward(self, gen_mel, real_mel):
        return super().forward(gen_mel, real_mel)
