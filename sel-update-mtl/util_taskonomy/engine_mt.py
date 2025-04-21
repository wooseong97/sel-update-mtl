import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util_taskonomy.misc as misc
# import util.lr_sched as lr_sched
import torch.nn.functional as F
from util_taskonomy.metric import *
from tqdm import tqdm


import pdb

def get_loss(outputs, targets, task):
    if 'class' in task:
        task_loss = F.mse_loss(outputs, targets.squeeze(1))
    elif 'segment_semantic' in task:
        criterion = torch.nn.CrossEntropyLoss()
        task_loss = criterion(outputs, targets)
    elif 'normal' in task:
        T = targets
        task_loss = (1 - (outputs*T).sum(-1) / (torch.norm(outputs, p=2, dim=-1) + 0.000001) / (torch.norm(T, p=2, dim=-1)+ 0.000001) ).mean()
    elif 'depth' in task or 'keypoint' in task or 'reshading' in task or 'edge' in task or 'segment' in task:
        if len(outputs.shape) == 4:
            Out = outputs.squeeze()
        task_loss = F.l1_loss(Out, targets)
    else:
        if len(outputs.shape) == 4:
            Out = outputs.squeeze()
        task_loss = F.mse_loss(Out, targets)
    return task_loss

def get_metric(outputs, targets, task):
    # get the metric
    if 'class' in task:
        metric = (outputs.argmax(dim=-1) == targets.argmax(dim=-1)).float().mean().item()
    elif 'depth' in task:
        if len(outputs.shape) == 4:
            outputs = outputs.squeeze()
        if task == 'depth_euclidean':
            metric = compute_depth_errors(outputs, targets).item()
        else:
            metric = 0.0
    elif 'curvature' in task:
        if len(outputs.shape) == 4:
            outputs = outputs.squeeze()
        metric = F.mse_loss(outputs, targets).item()
    else:
        if len(outputs.shape) == 4:
            outputs = outputs.squeeze()
        metric = F.l1_loss(outputs, targets).item()
    return metric



@torch.no_grad()
def evaluate(data_loader, model, device, AWL, p):
    metric_logger = misc.MetricLogger(delimiter="  ")
    model.eval()
    if AWL: AWL.eval()

    for i_data, data in enumerate(tqdm(data_loader)):
        samples = data['rgb']
        samples = samples.to(device, non_blocking=True)

        the_loss = {}
        loss_list = []
        the_metric = {}
        tot_loss = 0
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            for task in p.TASKS.NAMES:
                if 'rgb' in task:
                    continue
                targets = data[task].to(device, non_blocking=True)
                task_loss = get_loss(outputs[task], targets, task)
                tot_loss = tot_loss + task_loss.item()
                the_loss[task] = task_loss
                loss_list.append(the_loss[task])
                task_metric = get_metric(outputs[task], targets, task)
                the_metric[task] = task_metric

        if AWL: loss = AWL(loss_list)
        else: loss = sum(loss_list)

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(tot_loss=tot_loss)

        for _key, value in the_loss.items():
            metric_logger.meters[_key].update(value.item(), n=batch_size)

        for _key, value in the_metric.items():
            metric_logger.meters['met_'+_key].update(value, n=batch_size)

    print('test Result: ', ' '.join(str(a) + ':' + str(b.global_avg) for (a,b) in metric_logger.meters.items()))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}