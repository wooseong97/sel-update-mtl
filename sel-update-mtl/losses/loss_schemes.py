import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt, tasks):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in tasks}
        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in tasks]))
        return out



class MTLoss_affinity(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MTLoss_affinity, self).__init__()
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        self.group = None
    
    def forward(self, pred, gt, tasks):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in tasks}
        
        for group, comp in self.group.items():
            out[group] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in comp]))

        return out

    
from util_taskonomy.engine_mt import get_loss
class MTLoss_taskonomy(nn.Module):
    def __init__(self, p, tasks: list):
        super(MTLoss_taskonomy, self).__init__()
        self.p = p
        self.tasks = tasks

    def forward(self, pred, gt, tasks):
        loss_list = []
        for task in tasks: loss_list.append(get_loss(pred[task], gt[task], task))
        out = dict(zip(tasks, loss_list))
        out['total'] = sum(loss_list)
        return out


class MTLoss_affinity_taskonomy(nn.Module):
    def __init__(self, p, tasks: list):
        super(MTLoss_affinity_taskonomy, self).__init__()
        self.p = p
        self.tasks = tasks
        self.group = None
    
    def forward(self, pred, gt, tasks):
        loss_list = []
        for task in tasks: loss_list.append(get_loss(pred[task], gt[task], task))
        out = dict(zip(tasks, loss_list))
        
        for group, comp in self.group.items():
            out[group] = torch.sum(torch.stack([out[t] for t in comp]))

        return out
    
