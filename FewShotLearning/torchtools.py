from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

def open_all_layers(model):
    """
    Open all layers in model for training.
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def open_specified_layers(model, open_layers):
    """
    Open specified layers in model for training while keeping 
    other layers frozen.

    Args:
    - model (nn.Module): neural net model.
    - open_layers (list): list of layers names.
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    for layer in open_layers:
        assert hasattr(model, layer), "'{}' is not an attribute of the model, please provide the correct name".format(layer)

    for name, module in model.named_children():
        if name in open_layers:
            #print(module)
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False



def adjust_learning_rate(optimizer, iters, LUT):
    # decay learning rate by 'gamma' for every 'stepsize'
    for (stepvalue, base_lr) in LUT:
        if iters < stepvalue:
            lr = base_lr
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_no_margin(optimizer, iters, LUT):
    # decay learning rate by 'gamma' for every 'stepsize'
    for (stepvalue, base_lr) in LUT:
        if iters < stepvalue:
            lr = base_lr
            break
    n_para_groups  = len(optimizer.param_groups)
    for ii, param_group in enumerate(optimizer.param_groups):
        if ii < n_para_groups-1:
            param_group['lr'] = lr
    return lr


def adjust_lambda(iters, LUT):
    for (stepvalue, base_lambda) in LUT:
        if iters < stepvalue:
            lambda_xent = base_lambda
            break
    return lambda_xent


def set_bn_to_eval(m):
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def count_num_param(model):
    num_param = sum(p.numel() for p in model.parameters()) / 1e+06
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel() for p in model.classifier.parameters()) / 1e+06
    return num_param


def one_hot(labels, num_classes):
    """
    Turn the labels to one-hot encoding.
    Args:
        labels: [batch_size, num_examples]
    Return:
        labels_1hot: [batch_size, num_examples, num_classes]
    """
    labels_1hot_size = list(labels.size()) + [num_classes, ]
    labels_unsqueeze = labels.unsqueeze(-1)
    labels_1hot = torch.zeros(labels_1hot_size).scatter_(len(labels_1hot_size) - 1, labels_unsqueeze, 1)
    return labels_1hot


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      Args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: target learning rate = base lr * multiplier
          total_epoch: target learning rate is reached at total_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is not None:
            if epoch > self.total_epoch:
                self.after_scheduler.step(epoch - self.total_epoch)
            else:
                super(GradualWarmupScheduler, self).step(epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)



def get_scheduler(args, optimizer, n_iter_per_epoch):
    cosine_scheduler = CosineAnnealingLR(
        optimizer=optimizer, eta_min=0.000001,
        T_max=(args.stop_epoch - args.warmup_epoch) * n_iter_per_epoch)
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=args.warmup_multiplier,
        total_epoch=args.warmup_epoch * n_iter_per_epoch,
        after_scheduler=cosine_scheduler)
    return scheduler
