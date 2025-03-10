# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from spikingjelly.clock_driven import functional

import torch.nn.functional as F
from ZO_Estim.ZO_Estim_entry import build_obj_fn
from ZO_Estim.ZO_utils import default_create_bwd_pre_hook_ZO_grad

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None, ZO_Estim=None,
                    args=None):
    model.train(True)
    
    # Freeze all BatchNorm layers
    if getattr(args, 'freeze_bn', False):
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                module.eval()
                
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 2000

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        if ZO_Estim is not None:
            obj_fn = build_obj_fn(ZO_Estim.obj_fn_type, data=samples, target=targets, model=model, criterion=criterion)
            ZO_Estim.update_obj_fn(obj_fn)
            # with torch.no_grad():
            #     pred, loss = obj_fn()
            #     ZO_Estim.estimate_grad(old_loss=loss)
            
            outputs, loss = ZO_Estim.estimate_grad()
                
            ### pseudo NP
            if ZO_Estim.splited_layer_list is not None:
                # bwd_pre_hook_list = []
                # for splited_layer in ZO_Estim.splited_layer_list:
                #     create_bwd_pre_hook_ZO_grad = getattr(splited_layer.layer, 'create_bwd_pre_hook_ZO_grad', default_create_bwd_pre_hook_ZO_grad)
                #     bwd_pre_hook_list.append(splited_layer.layer.register_full_backward_pre_hook(create_bwd_pre_hook_ZO_grad(splited_layer.layer.ZO_grad_output, args.debug)))
                # output = model(data)
                # loss = criterion(output, target)
                # loss.backward()
                
                # for bwd_pre_hook in bwd_pre_hook_list:
                #     bwd_pre_hook.remove()
                
                fwd_hook_list = []
                for splited_layer in ZO_Estim.splited_layer_list:

                    fwd_hook_get_param_grad = splited_layer.layer.create_fwd_hook_get_param_grad(splited_layer.layer.ZO_grad_output, args.debug)
                    fwd_hook_list.append(splited_layer.layer.register_forward_hook(fwd_hook_get_param_grad))
                    
                    with torch.no_grad():
                        output = model(samples)
                        loss = criterion(output, targets)
                
                for fwd_hook_handle in fwd_hook_list:
                    fwd_hook_handle.remove()
            
            ### save param FO grad
            if args.debug:
                for param in model.parameters():
                    if param.requires_grad:
                        param.ZO_grad = param.grad.clone()
                        
                optimizer.zero_grad()
                
                output = model(samples)
                loss = criterion(output, targets)
                loss.backward()
                
                for param in model.parameters():
                    if param.requires_grad:
                        param.FO_grad = param.grad.clone()
                
                optimizer.zero_grad()
            
            ### print FO ZO grad
                print('param cos sim')
                for param in model.parameters():
                    if param.requires_grad:
                        print(f'{F.cosine_similarity(param.FO_grad.view(-1), param.ZO_grad.view(-1), dim=0)}')
                    
                print('param Norm ZO/FO: ')
                for param in model.parameters():
                    if param.requires_grad:
                        print(f'{torch.linalg.norm(param.ZO_grad.view(-1)) / torch.linalg.norm(param.FO_grad.view(-1))}')
                
                optimizer.zero_grad()
            
            loss_value = loss.item()
            optimizer.step()
            
        else:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss = loss / accum_iter
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()
        functional.reset_net(model)
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 500, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        functional.reset_net(model)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
