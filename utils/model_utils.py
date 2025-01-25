import random
import torch
import numpy as np
from torch.nn.utils import weight_norm

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def setup_seed(seed):
    random.seed(seed)                
    np.random.seed(seed)
    torch.manual_seed(seed)        
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def closest_power_of_two(n):
    return 1 << (n - 1).bit_length()
