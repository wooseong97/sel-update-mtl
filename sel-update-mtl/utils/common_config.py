import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil

from configs.mypath import PROJECT_ROOT_DIR
from configs.mypath import db_root_taskonomy
import os

import pdb


def get_backbone(p):
    """ Return the backbone """    
    if p['backbone'] == 'vitT':
        from models.transformers.vit import vit_tiny_patch16_384
        backbone = vit_tiny_patch16_384(pretrained=True, drop_path_rate=0.15, img_size=p.TRAIN.SCALE)
        backbone_channels = [192 for _ in range(4)]
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0]//16, p.TRAIN.SCALE[1]//16] for _ in range(4)]
    else:
        raise NotImplementedError

    return backbone, backbone_channels


def get_head(p, backbone_channels, task):
    """ Return the decoder head """
    if p['head'] == 'mlp':
        from models.transformers.transformer_decoder import MLPHead
        return MLPHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])

    else:
        raise NotImplementedError


def get_model(p):
    """ Return the model """

    backbone, backbone_channels = get_backbone(p)
    
    if p['model'] == 'TransformerMTL':
        from models.transformer_net import TransformerMTL
        feat_channels = p.backbone_channels[-1]
        heads = torch.nn.ModuleDict({task: get_head(p, feat_channels, task) for task in p.TASKS.NAMES})
        model = TransformerMTL(p, backbone, backbone_channels, heads)
    elif p['model'] == 'TransformerMTL_multi':
        from models.transformer_net import TransformerMTL_multi
        feat_channels = sum(p.backbone_channels)
        heads = torch.nn.ModuleDict({task: get_head(p, feat_channels, task) for task in p.TASKS.NAMES})
        model = TransformerMTL_multi(p, backbone, backbone_channels, heads)
    else:
        raise NotImplementedError('Unknown model {}'.format(p['model']))
    
    return model


"""
    Transformations, datasets and dataloaders
"""
def get_transformations(p):
    """ Return transformations for training and evaluationg """
    from data import transforms
    import torchvision

    # Training transformations
    if p['train_db_name'] == 'NYUD' or p['train_db_name'] == 'PASCALContext':
        train_transforms = torchvision.transforms.Compose([ # from ATRC
            # transforms.Resize(size=p.TRAIN.SCALE),                    
            transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=p.TRAIN.SCALE, cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TRAIN.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])

        # Testing 
        valid_transforms = torchvision.transforms.Compose([
            # transforms.Resize(size=p.TEST.SCALE),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TEST.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        return train_transforms, valid_transforms

    else:
        return None, None


def get_train_dataset(p, transforms=None):
    """ Return the train dataset """

    db_name = p['train_db_name']
    print('Preparing train dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p.db_paths['PASCALContext'], download=False, split=['train'], transform=transforms, retname=True,
                                          do_semseg='semseg' in p.TASKS.NAMES,
                                          do_edge='edge' in p.TASKS.NAMES,
                                          do_normals='normals' in p.TASKS.NAMES,
                                          do_sal='sal' in p.TASKS.NAMES,
                                          do_human_parts='human_parts' in p.TASKS.NAMES,
                                          overfit=False)

    if db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_paths['NYUD_MT'], download=False, split='train', transform=transforms, do_edge='edge' in p.TASKS.NAMES, 
                                    do_semseg='semseg' in p.TASKS.NAMES, 
                                    do_normals='normals' in p.TASKS.NAMES, 
                                    do_depth='depth' in p.TASKS.NAMES, overfit=False)
        
        
    if db_name == 'Taskonomy':
        from util_taskonomy.dataset_taskonomy import TaskonomyDataset
        img_types = ['depth_euclidean', 'depth_zbuffer', 'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d', 'normal', 'principal_curvature', 'reshading', 'rgb', 'segment_unsup2d', 'segment_unsup25d']
        database = TaskonomyDataset(img_types, data_dir=db_root_taskonomy, split='tiny', partition='train', resize_scale=256, crop_size=256, fliplr=True)

    return database


def get_train_dataloader(p, dataset, sampler):
    """ Return the train dataloader """
    collate = collate_mil
    trainloader = DataLoader(dataset, batch_size=p['trBatch'], drop_last=True,
                             num_workers=p['nworkers'], collate_fn=collate, pin_memory=True, sampler=sampler)
    return trainloader


def get_test_dataset(p, transforms=None):
    """ Return the test dataset """

    db_name = p['val_db_name']
    print('Preparing test dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p.db_paths['PASCALContext'], download=False, split=['val'], transform=transforms, retname=True,
                                      do_semseg='semseg' in p.TASKS.NAMES,
                                      do_edge='edge' in p.TASKS.NAMES,
                                      do_normals='normals' in p.TASKS.NAMES,
                                      do_sal='sal' in p.TASKS.NAMES,
                                      do_human_parts='human_parts' in p.TASKS.NAMES,
                                    overfit=False)
    
    elif db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_paths['NYUD_MT'], download=False, split='val', transform=transforms, do_edge='edge' in p.TASKS.NAMES, 
                                    do_semseg='semseg' in p.TASKS.NAMES, 
                                    do_normals='normals' in p.TASKS.NAMES, 
                                    do_depth='depth' in p.TASKS.NAMES)
        
    elif db_name == 'Taskonomy':
        from util_taskonomy.dataset_taskonomy import TaskonomyDataset
        img_types = ['depth_euclidean', 'depth_zbuffer', 'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d', 'normal', 'principal_curvature', 'reshading', 'rgb', 'segment_unsup2d', 'segment_unsup25d']
        database = TaskonomyDataset(img_types, data_dir=db_root_taskonomy, split='tiny', partition='test', resize_scale=256, crop_size=256)

    else:
        raise NotImplemented("test_db_name: Choose among PASCALContext and NYUD")

    return database


def get_test_dataloader(p, dataset):
    """ Return the validation dataloader """
    collate = collate_mil
    testloader = DataLoader(dataset, batch_size=p['valBatch'], shuffle=False, drop_last=False,
                            num_workers=p['nworkers'], pin_memory=True, collate_fn=collate)
    return testloader


""" 
    Loss functions 
"""
def get_loss(p, task=None):
    """ Return loss function for a specific task """

    if task == 'edge':
        from losses.loss_functions import BalancedBinaryCrossEntropyLoss
        criterion = BalancedBinaryCrossEntropyLoss(pos_weight=p['edge_w'], ignore_index=p.ignore_index)

    elif task == 'semseg' or task == 'human_parts':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(ignore_index=p.ignore_index)

    elif task == 'normals':
        from losses.loss_functions import L1Loss
        criterion = L1Loss(normalize=True, ignore_index=p.ignore_index)

    elif task == 'sal':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(balanced=True, ignore_index=p.ignore_index) 

    elif task == 'depth':
        from losses.loss_functions import L1Loss
        criterion = L1Loss()

    else:
        criterion = None

    return criterion


def get_criterion(p):
    if p['train_db_name']=='Taskonomy':
        if 'loss_schema' in p.keys() and p['loss_schema'] in ['affinity']:
            from losses.loss_schemes import MTLoss_affinity_taskonomy
            return MTLoss_affinity_taskonomy(p, p.TASKS.NAMES)
        else:
            from losses.loss_schemes import MTLoss_taskonomy
            return MTLoss_taskonomy(p, p.TASKS.NAMES)
    elif 'loss_schema' in p.keys() and p['loss_schema'] in ['affinity']:
        from losses.loss_schemes import MTLoss_affinity
        loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
        loss_weights = p['loss_kwargs']['loss_weights']
        return MTLoss_affinity(p, p.TASKS.NAMES, loss_ft, loss_weights)
    else:
        from losses.loss_schemes import MultiTaskLoss
        loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
        loss_weights = p['loss_kwargs']['loss_weights']
        return MultiTaskLoss(p, p.TASKS.NAMES, loss_ft, loss_weights)


"""
    Optimizers and schedulers
"""
def get_optimizer(p, model, criterion=None):
    """ Return optimizer for a given model and setup """

    if p['train_db_name']=='Taskonomy':
        print('Optimizer for Taskonomy dataset')
        if 'lr_ts' in p.keys() and p['lr_ts']:
            print('Optimizer uses two parameter groups')
            params = [{'params': get_lr_params(model, p=p, part='shared', tasks=p.TASKS.NAMES)},
                    {'params': get_lr_params(model, p=p, part='task_specific', tasks=p.TASKS.NAMES),
                     'lr': float(p['optimizer_kwargs']['lr'])*len(p.TASKS.NAMES)},]
        else:
            print('Optimizer uses a single parameter group - (Default)')
            params = [{'params': model.parameters()},]

    elif 'lr_ts' in p.keys() and p['lr_ts']:
        print('Optimizer uses two parameter groups')
        params = [{'params': get_lr_params(model, p=p, part='shared', tasks=p.TASKS.NAMES)},
                {'params': get_lr_params(model, p=p, part='task_specific', tasks=p.TASKS.NAMES), 'lr': float(p['optimizer_kwargs']['lr'])*len(p.TASKS.NAMES)}]
    else:
        print('Optimizer uses a single parameter group - (Default)')
        params = model.parameters()

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])
    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    # get scheduler    
    if p.scheduler == 'poly':
        from utils.train_utils import PolynomialLR
        scheduler = PolynomialLR(optimizer, p.max_iter, gamma=0.9, min_lr=0)

    return scheduler, optimizer



def get_lr_params(model, p=None, part=None, tasks=None):
    def ismember(comp, tasks):
        exists = False
        for task in tasks:
            exists = exists or comp.find(task) > 0
        return exists
    
    if hasattr(model, 'module'):
        model = model.module
    
    if part == 'task_specific':
        b = [model]
    elif part == 'shared':
        b = [model]

    for i in range(len(b)):
        for name, k in b[i].named_parameters():
            if k.requires_grad:
                if part == 'task_specific':
                    if ismember(name, tasks):
                        yield k
                elif part == 'shared':
                    if ismember(name, tasks):
                        continue
                    else:
                        yield k
