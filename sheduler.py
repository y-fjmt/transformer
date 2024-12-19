import torch
from torch.optim.lr_scheduler import _LRScheduler

class CustomLRScheduler(_LRScheduler):
    
    def __init__(
            self, 
            optimizer: torch.optim.Optimizer,
            d_model: int = 512, 
            warmup_steps: int = 4000,
            last_epoch=-1
        ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = max(1, self.last_epoch + 1)
        lrate = (self.d_model ** (-0.5)) * min(step_num ** (-0.5), step_num * (self.warmup_steps**(-1.5)))
        return [lrate for _ in self.optimizer.param_groups]