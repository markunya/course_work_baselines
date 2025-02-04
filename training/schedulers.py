from typing import Literal
from utils.class_registry import ClassRegistry
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR

schedulers_registry = ClassRegistry()

ReduceLrTime = Literal['epoch', 'step', 'period']

class WarmUpScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, base_scheduler, **kwargs):
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.finished_warmup = False
        self.current_step = 0

        def lr_lambda(current_step):
            if current_step <= warmup_steps:
                if current_step == warmup_steps:
                    self.finished_warmup = True
                return current_step / warmup_steps

        super().__init__(optimizer, lr_lambda, **kwargs)

    def step(self, epoch=None):
        if self.finished_warmup:
            self.base_scheduler.step(epoch)
        else:
            self.current_step += 1
            super().step(epoch)

    def get_last_lr(self):
        if self.finished_warmup:
            return self.base_scheduler.get_last_lr()
        return super().get_last_lr()
    
class BaseScheduler:
    def __init__(self, optimizer, base_scheduler_type,
                reduce_time: ReduceLrTime, step_period=None, warmup_steps=0, **kwargs):
        base_scheduler = base_scheduler_type(optimizer, **kwargs)
        self.reduce_time = reduce_time
        if reduce_time == 'period':
            self.period = step_period

        if warmup_steps > 0:
            self.scheduler = WarmUpScheduler(optimizer, warmup_steps, base_scheduler)
        else:
            self.scheduler = base_scheduler

    def step(self, epoch=None):
        self.scheduler.step(epoch)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

@schedulers_registry.add_to_registry(name='multi_step')
class MultiStepScheduler(BaseScheduler):
    def __init__(self, optimizer, reduce_time: ReduceLrTime, step_period=None, warmup_steps=0, **kwargs):
        super().__init__(optimizer, MultiStepLR, reduce_time, step_period, warmup_steps, **kwargs)

@schedulers_registry.add_to_registry(name='exponential')
class ExponentialScheduler(BaseScheduler):
    def __init__(self, optimizer, reduce_time: ReduceLrTime, step_period=None, warmup_steps=0, **kwargs):
        super().__init__(optimizer, ExponentialLR, reduce_time, step_period, warmup_steps, **kwargs)
