from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from .mot import MotTrainer

train_factory = {
    'mot': MotTrainer,
}


def get_cosine_schedule_with_warmup(
        learning_rate, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5,
        min_lr: float = 1e-5, last_epoch: int = 0
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        learning_rate (:obj:`float`):
            The base learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        min_lr (:obj:`float`, `optional`, defaults to 0.5):
            The minimum of learning rate.
        last_epoch (:obj:`int`, `optional`, defaults to 0):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    lrs = {}
    for current_step in range(last_epoch + 1, num_training_steps + 1):
        cur_lr = lr_lambda(current_step) * learning_rate
        lrs[current_step] = (cur_lr + min_lr)
    return lrs
