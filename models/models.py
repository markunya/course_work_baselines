import torch.functional as F
import torch.nn.functional as F
from utils.class_registry import ClassRegistry
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from utils.model_utils import init_weights, get_padding
from torch import nn
from typing import Literal

models_registry = ClassRegistry()

LRELU_SLOPE = 0.1

class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5),
                norm_type: Literal["weight", "spectral"] = "weight"):
        super(ResBlock1, self).__init__()
        self.norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.convs1 = nn.ModuleList([
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                            padding=get_padding(kernel_size, dilation[0]))),
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                            padding=get_padding(kernel_size, dilation[1]))),
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                            padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                            padding=get_padding(kernel_size, 1))),
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                            padding=get_padding(kernel_size, 1))),
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                            padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        if self.norm != weight_norm:
            raise RuntimeWarning("No weight norm to remove")
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3),
                 norm_type: Literal["weight", "spectral"] = "weight"):
        super(ResBlock2, self).__init__()
        self.norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.convs = nn.ModuleList([
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                            padding=get_padding(kernel_size, dilation[0]))),
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                            padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        if self.norm != weight_norm:
            raise RuntimeWarning("No weight norm to remove")
        for l in self.convs:
            remove_weight_norm(l)
