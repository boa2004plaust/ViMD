# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import timm
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from losses import DistillKL, CELoss, HSMSE


def train_one_epoch(sr_model: torch.nn.Module, model: torch.nn.Module, model_hr: torch.nn.Module,
                    criterion_ce: CELoss, criterion_logitkd: DistillKL, criterion_fd: HSMSE,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args=None):
    sr_model.eval()
    model_hr.eval()
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # debug
    # count = 0
    for samples_hr, samples_lr, targets in metric_logger.log_every(data_loader, print_freq, header):
        # count += 1
        # if count > 20:
        #     break

        samples_hr = samples_hr.to(device, non_blocking=True)
        samples_lr = samples_lr.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with amp_autocast():
            sr_outputs = sr_model(samples_lr)
            hidden_states_features_lr, outputs = model(sr_outputs,
                                                       if_random_cls_token_position=args.if_random_cls_token_position,
                                                       if_random_token_rank=args.if_random_token_rank)
            # outputs = model(samples)
            loss = criterion_ce(outputs, targets)
            with torch.no_grad():
                hidden_states_features_hr, hr_outputs = model_hr(samples_hr)
            # loss += 1.*criterion_logitkd(outputs, hr_outputs)
            loss += 5. * criterion_fd(hidden_states_features_lr, hidden_states_features_hr)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, sr_model, model, device, amp_autocast):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    sr_model.eval()
    model.eval()

    for samples_hr, samples_lr, targets in metric_logger.log_every(data_loader, 10, header):
        images = samples_lr.to(device, non_blocking=True)
        target = targets.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            sr_output = sr_model(images)
            hidden_states_features_lr, output = model(sr_output)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
