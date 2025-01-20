from utils.class_registry import ClassRegistry
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ExponentialLR

schedulers_registry = ClassRegistry()

@schedulers_registry.add_to_registry(name='multi_step')
class MultiStep(MultiStepLR):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

@schedulers_registry.add_to_registry(name='exponential')
class Exponential(ExponentialLR):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
