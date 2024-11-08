import collections
import datetime
import glob
import numpy as np
import os
import pandas as pd
import random
import re
from tqdm.autonotebook import tqdm

import torch
from torch.utils.data.dataloader import default_collate
from monai.networks.utils import one_hot

def list_data_collate(batch: collections.abc.Sequence):
    """
    Combine instances from a list of dicts into a single dict, by stacking them along first dim
    [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
    followed by the default collate which will form a batch BxNx3xHxW
    """

    for i, item in enumerate(batch):
        # print(f"{i} = {item['image'].shape=} >> {item['image'].keys=}")
        data = item[0]
        data["image"] = torch.stack([ix["image"] for ix in item], dim=0)
        # data["patch_location"] = torch.stack([ix["patch_location"] for ix in item], dim=0)
        batch[i] = data
    return default_collate(batch)


def mrff_to_grade(x):
    if x<5:
        return 0
    elif x<15:
        return 1
    elif x<25:
        return 2
    else:
        return 3
    
class PILImage_Convert:
    def __init__(self, convert_str: str = "RGB"):
        self.convert_str = convert_str
    def __call__(self, x):
        return x.convert(self.convert_str)
    
# Copied from: https://github.com/HiLab-git/SSL4MIS/blob/master/code/utils/ramps.py
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_rampup_weight(t, t_max):
    return sigmoid_rampup(t, t_max)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha = 1 - alpha)
        
        
def tuple_index(x, idx):
    if isinstance(x, tuple):
        return tuple(xx[idx] for xx in x)
    else:
        return x[idx]
    
    
class OrdinalConvert(object):
    def __init__(self, range_min = 0, range_max = 100, bin_number=100, sigma=2):
        bin_size = (range_max - range_min)/bin_number        
        range_min = range_min - 0.5*bin_size
        range_max = range_max - 0.5*bin_size        
        bin_ticks = torch.linspace(range_min,range_max,bin_number+1)
        bins = 0.5*(bin_ticks[:-1] + bin_ticks[1:])
        
        self.range_min = range_min
        self.range_max = range_max
        self.bin_number = bin_number
        self.sigma = sigma
        self.bins = bins
        self.bin_ticks = bin_ticks
        self.bin_size = bin_size
        
    def convert_x_to_class(self, x):
        assert len(x.shape)==2
        _bin_ticks = self.bin_ticks.to(x.device)
        x = torch.clamp(x, self.range_min, self.range_max)
        x_cls = torch.stack([torch.argwhere((a.ge(_bin_ticks))[:-1]).max() for a in x]).unsqueeze(1)
        return x_cls
    
    def convert_x_to_onehot(self, x):
        x_bin = self.convert_x_to_class(x)
        return one_hot(x_bin, self.bin_number)
    
    def convert_proba_to_x(self, x):
        _bins = self.bins.to(x.device)
        return torch.sum(x * _bins, 1, keepdim=True)
    
    def convert_x_to_proba_normal(self, x, sigma: float=None):
        _sh = (len(x), self.bin_number)
        _y = self.bins.to(x.device)
        _y = _y.broadcast_to(_sh)
        _x = x.broadcast_to(_sh)
        
        if sigma is None:
            sigma = self.sigma
        _sigma = torch.ones_like(x, device=x.device)*sigma
        _sigma = _sigma.broadcast_to(_sh)
        
        proba = torch.exp(-((_y - _x)**2)/(2*_sigma**2)) / (torch.sqrt(2*torch.pi*_sigma))
        proba = proba / torch.sum(proba, dim=1, keepdim=True).broadcast_to(proba.shape)
        return proba
    
    def convert_x_to_proba_sord_l1(self, x):
        _sh = (len(x), self.bin_number)
        _y = self.bins.to(x.device)
        _y = _y.broadcast_to(_sh)
        _x = x.broadcast_to(_sh)
        
        proba = - torch.abs(_y - _x)
        proba = torch.softmax(proba, dim=1)
        return proba
    
    def convert_x_to_proba_sord_l2(self, x):
        _sh = (len(x), self.bin_number)
        _y = self.bins.to(x.device)
        _y = _y.broadcast_to(_sh)
        _x = x.broadcast_to(_sh)
        
        proba = - (_y - _x)**2
        proba = torch.softmax(proba, dim=1)
        return proba
    
    
class BoneAgeLabelText(object):
    dict_male = {
        0: "Standard bone age : 3 Months.\nNormal range : -6M",
        1: "Standard bone age : 6 Months.\nNormal range : 2-17M",
        2: "Standard bone age : 9 Months.\nNormal range : 2-17M",
        3: "Standard bone age : 11.5 Months.\nNormal range : 5-18M",
        4: "Standard bone age : 15 Months.\nNormal range : 8-19M",
        5: "Standard bone age : 18 Months.\nNormal range : 9-20M",
        6: "Standard bone age : 21 Months.\nNormal range : 11-27M",
        7: "Standard bone age : 23 Months.\nNormal range : 22-32M",
        8: "Standard bone age : 27 Months.\nNormal range : 22-32M",
        9: "Standard bone age : 30 Months.\nNormal range : 23-38M",
        10: "Standard bone age : 36 Months.\nNormal range : 26-42M",
        11: "Standard bone age : 41 Months.\nNormal range : 38-56M",
        12: "Standard bone age : 48 Months.\nNormal range : 39-60M",
        13: "Standard bone age : 53 Months.\nNormal range : 40-66M",
        14: "Standard bone age : 59 Months.\nNormal range : 44-70M",
        15: "Standard bone age : 66 Months.\nNormal range : 57-75M",
        16: "Standard bone age : 73 Months.\nNormal range : 66-88M",
        17: "Standard bone age : 85 Months.\nNormal range : 75-96M",
        18: "Standard bone age : 96 Months.\nNormal range : 79-115M",
        19: "Standard bone age : 109 Months.\nNormal range : 103-122M",
        20: "Standard bone age : 120 Months.\nNormal range : 106-137M",
        21: "Standard bone age : 131 Months.\nNormal range : 115-143M",
        22: "Standard bone age : 142 Months.\nNormal range : 138-150M",
        23: "Standard bone age : 157 Months.\nNormal range : 151-180M",
        24: "Standard bone age : 170 Months.\nNormal range : 153-186M",
        25: "Standard bone age : 180 Months.\nNormal range : 153M-",
        26: "Standard bone age : 192 Months.\nNormal range : 167M-",
    }

    dict_female = {
        0: "Standard bone age : 2 Months.\nNormal range : -4M",
        1: "Standard bone age : 6 Months.\nNormal range : 2-10M",
        2: "Standard bone age : 9 Months.\nNormal range : 2-10M",
        3: "Standard bone age : 12 Months.\nNormal range : 10-15M",
        4: "Standard bone age : 15 Months.\nNormal range : 11-17M",
        5: "Standard bone age : 18 Months.\nNormal range : 15-27M",
        6: "Standard bone age : 21 Months.\nNormal range : 15-27M",
        7: "Standard bone age : 24 Months.\nNormal range : 15-28M",
        8: "Standard bone age : 27 Months.\nNormal range : 18-35M",
        9: "Standard bone age : 30 Months.\nNormal range : 28-43M",
        10: "Standard bone age : 36 Months.\nNormal range : 29-45M",
        11: "Standard bone age : 41 Months.\nNormal range : 32-49M",
        12: "Standard bone age : 48 Months.\nNormal range : 33-50M",
        13: "Standard bone age : 53 Months.\nNormal range : 40-65M",
        14: "Standard bone age : 60 Months.\nNormal range : 51-75M",
        15: "Standard bone age : 65.5 Months.\nNormal range : 55-77M",
        16: "Standard bone age : 72.5 Months.\nNormal range : 63-87M",
        17: "Standard bone age : 86 Months.\nNormal range : 66-89M",
        18: "Standard bone age : 95 Months.\nNormal range : 81-102M",
        19: "Standard bone age : 108 Months.\nNormal range : 103-124M",
        20: "Standard bone age : 122.5 Months.\nNormal range : 105-126M",
        21: "Standard bone age : 132 Months.\nNormal range : 127-145M",
        22: "Standard bone age : 144 Months.\nNormal range : 139-155M",
        23: "Standard bone age : 157 Months.\nNormal range : 144-169M",
        24: "Standard bone age : 169 Months.\nNormal range : 146-187M",
        25: "Standard bone age : 180 Months.\nNormal range : 161M-",
        26: "Standard bone age : 192 Months.\nNormal range : 161M-",
    }
    
    boneage_male = [float(re.findall(r"\d+[.,]?\d*", x)[0]) for x in dict_male.values()]
    boneage_female = [float(re.findall(r"\d+[.,]?\d*", x)[0]) for x in dict_female.values()]

    def get_labeltext(self, x: int, female: bool, text: bool=True):
        if female:
            if text:
                return self.dict_female[x]
            else:
                return self.boneage_female[x]
        else:
            if text:
                return self.dict_male[x]
            else:
                return self.boneage_male[x]
    
def get_boneage_labeltext(x: int, female: bool, text: bool=True):    
    if female:
        if text:
            return BoneAgeLabelText.dict_female[x]
        else:
            return BoneAgeLabelText.boneage_female[x]
    else:
        if text:
            return BoneAgeLabelText.dict_male[x]
        else:
            return BoneAgeLabelText.boneage_male[x]


def convert_coral_levels_to_proba(x: torch.Tensor, dim: int=1):
    t0 = torch.zeros_like(x).argmax(dim, keepdims=True)
    t1 = torch.zeros_like(x).argmax(dim, keepdims=True) + 1
    tx = torch.cat([t1, x, t0], dim)
    _idx0 = [slice(None) for _ in range(dim+1)]
    _idx1 = [slice(None) for _ in range(dim+1)]
    _idx0[dim] = slice(0,-1)
    _idx1[dim] = slice(1,None)
    _idx0 = tuple(_idx0)
    _idx1 = tuple(_idx1)

    ty = tx[_idx0] - tx[_idx1]
    return ty

def convert_proba_to_coral_levels(x: torch.Tensor, dim: int=1):
    tx = 1 - torch.cumsum(x, dim=dim)
    _idx = [slice(None) for _ in range(dim+1)]
    _idx[dim] = slice(None, tx.shape[dim] - 1)
    _idx = tuple(_idx)
    return tx[_idx]
