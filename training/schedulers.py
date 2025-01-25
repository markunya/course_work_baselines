from enum import Enum
from utils.class_registry import ClassRegistry
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ExponentialLR

schedulers_registry = ClassRegistry()

class ReduceLrOnEach(Enum):
    step = 0,
    epoch = 1

@schedulers_registry.add_to_registry(name='multi_step')
class MultiStep(MultiStepLR):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.reduce_time = ReduceLrOnEach.step

@schedulers_registry.add_to_registry(name='exponential')
class Exponential(ExponentialLR):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.reduce_time = ReduceLrOnEach.epoch
