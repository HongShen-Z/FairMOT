import torch
import numpy as np


class GradNorm():

    def __init__(self, task_names, alpha=1.5):
        self._len = len(task_names)
        self._task_names = task_names
        self._alpha = alpha

        # init weights
        self._weights = torch.ones([len(task_names)])
        self._weights = torch.nn.Parameter(self._weights)
        # set L(0)
        self._l0 = None

    def _get_norm(self, grad):
        return (torch.sum(torch.mul(grad, grad))) ** 0.5

    def _get_weights(self):
        return self._weights

    def __call__(self, shared_w, losses):
        """
            Get the grad norm loss as well as weighted loss
            input:
                losses: dict, {task_name, loss}
                shared_w: last shared layer out tensor
            return:
                gnorm_loss: dict, {task_name, loss} loss after gradient_norm
                new_losses:
        """
        # weight和task name对应上
        net_losses = [losses[name] for name in self._task_names]

        if self._l0 is None:
            self._l0 = net_losses

        # task learning rate
        loss_rate = torch.stack(net_losses, dim=0) / torch.stack(self._l0, dim=0)
        loss_rate_mean = torch.mean(loss_rate)
        loss_r = loss_rate / loss_rate_mean

        # weight re-normalize
        factor = torch.div(torch.tensor(self._len, dtype=tf.float32), torch.sum(self._weights))
        weights = torch.mul(self._weights, factor)

        # grad loss
        grads = [torch.autograd.grad(loss, shared_w, retain_graph=True, create_graph=True)[0] for loss in net_losses]
        # grads = [tf.concat(gs, axis=1) for gs in grads]
        gnorms = [self._get_norm(g) for g in grads]
        gnorms = torch.stack(gnorms, dim=0)  # [T, ]
        avgnorm = torch.mean(gnorms)
        grad_diff = torch.abs(gnorms - (avgnorm * (loss_r ** self._alpha)).detach())
        gnorm_loss = torch.sum(grad_diff)
        weighted_losses = {}

        for i, name in enumerate(self._task_names):
            weighted_losses[name] = net_losses[i] * (weights[i]).detach()

        new_net_loss = sum(list(weighted_losses.values()))
        return gnorm_loss, new_net_loss


def call_gradnorm(shared_w, task_losses):
    """
        :param shared_w: last shared layer out tensor
        :param task_losses: dict, {task_name, loss}
        :return: loss, the sum of net loss and grad loss
    """
    task_names = [name for name in task_losses]
    gradnorm = GradNorm(task_names)
    gnorm_loss, net_loss = gradnorm(shared_w=shared_w, losses=task_losses)
    loss = gnorm_loss + net_loss
    return loss
