import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from easydict import EasyDict as edict
from einops import rearrange as o_rearrange
INTERPOLATE_MODE = 'bilinear'


def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()

class TransformerMTL(nn.Module):
    def __init__(self, p, backbone, backbone_channels, heads):
        super(TransformerMTL, self).__init__()
        
        self.p = p
        self.tasks = p.TASKS.NAMES
        self.backbone = backbone
        self.heads = heads 

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}

        # Backbone
        x, selected_fea = self.backbone(x) 
                
        oh, ow = self.p.spatial_dim[-1]
        _x = rearrange(x, 'b (h w) c -> b c h w', h=oh, w=ow)

        # Generate predictions
        for t in self.tasks: out[t] = F.interpolate(self.heads[t](_x), img_size, mode=INTERPOLATE_MODE)
            
        return out
    

class TransformerMTL_multi(nn.Module):
    def __init__(self, p, backbone, backbone_channels, heads):
        super(TransformerMTL_multi, self).__init__()
        
        self.p = p
        self.tasks = p.TASKS.NAMES
        self.backbone = backbone
        # self.multi_task_decoder = TransformerDecoder(p) 
        self.heads = heads 

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}

        # Backbone
        x, selected_fea = self.backbone(x)
        
        _x_list = []
        for i, fea in enumerate(selected_fea):
            oh, ow = self.p.spatial_dim[-1]
            _x = rearrange(fea, 'b (h w) c -> b c h w', h=oh, w=ow)
            _x_list.append(_x)
        
        _x = torch.cat(_x_list, dim=1)

        # Generate predictions
        for t in self.tasks: out[t] = F.interpolate(self.heads[t](_x), img_size, mode=INTERPOLATE_MODE)
            
        return out