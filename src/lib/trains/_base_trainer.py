from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
from torch.autograd import Variable
from trains.min_norm_solvers import MinNormSolver, gradient_normalizers


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss_fn, optimizer, tasks):
        super(ModleWithLoss, self).__init__()
        self.tasks = tasks
        self.loss_D = loss_fn['D']
        self.loss_R = loss_fn['R']
        self.loss = {'D': self.loss_D, 'R': self.loss_R}
        self.optimizer = optimizer
        self.model_rep = model['rep']
        self.model_D = model['D']
        self.model_R = model['R']
        self.model = {'rep': self.model_rep, 'D': self.model_D, 'R': self.model_R}
        self.loss_data = {}
        self.grads = {}
        self.scale = {}

    def forward(self, batch, phase='train'):
        loss_data = self.loss_data
        grads = self.grads
        scale = self.scale
        images = batch['input']
        images = Variable(images.cuda())

        self.optimizer.zero_grad()
        # First compute representations (z)
        with torch.no_grad():
            images_volatile = Variable(images.data)
            rep = self.model['rep'](images_volatile)
        # As an approximate solution we only need gradients for input
        if isinstance(rep, list):
            # This is a hack to handle psp-net
            rep = rep[0]
            rep_variable = [Variable(rep.data.clone(), requires_grad=True)]
            list_rep = True
        else:
            rep_variable = Variable(rep.data.clone(), requires_grad=True)
            list_rep = False

        # Compute gradients of each loss function wrt z
        for t in self.tasks:
            self.optimizer.zero_grad()
            out_t = self.model[t](rep_variable)
            loss, _ = self.loss[t](out_t, batch)
            loss_data[t] = loss.item()
            if phase == 'train':
                loss.backward()
            grads[t] = []
            if list_rep:
                grads[t].append(Variable(rep_variable[0].grad.data.clone(), requires_grad=False))
                rep_variable[0].grad.data.zero_()
            else:
                grads[t].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
                rep_variable.grad.data.zero_()

        # Normalize all gradients, this is optional and not included in the paper.
        gn = gradient_normalizers(grads, loss_data, 'loss+')
        for t in self.tasks:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        # Frank-Wolfe iteration to compute scales.
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in self.tasks])
        for i, t in enumerate(self.tasks):
            scale[t] = float(sol[i])

        # Scaled back-propagation
        self.optimizer.zero_grad()
        rep = self.model['rep'](images)
        outputs = {}
        loss_stats = {}
        for i, t in enumerate(self.tasks):
            out_t = self.model[t](rep)
            loss_t, loss_stat = self.loss[t](out_t, batch)
            outputs.update(out_t)
            loss_stats.update(loss_stat)
            loss_data[t] = loss_t.item()
            if i > 0:
                loss = loss + scale[t] * loss_t
            else:
                loss = scale[t] * loss_t
        loss_stats.update({'loss': loss})
        if phase == 'train':
            loss.backward()
            self.optimizer.step()

        # outputs = self.model(batch['input'])
        # loss, loss_stats = self.loss(outputs, batch)
        return outputs, loss, loss_stats


class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModleWithLoss(model, self.loss, self.optimizer, opt.tasks)
        loss_params = []
        for m in self.loss:
            loss_params += self.loss[m].parameters()
        self.optimizer.add_param_group({'params': loss_params})

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            gpus = list(range(len(gpus)))
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch, phase=phase)
            # loss = loss.mean()
            # if phase == 'train':
            #     # 2. add set_to_none=True
            #     self.optimizer.zero_grad()
            #     loss.backward()
            #     self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats, batch

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
