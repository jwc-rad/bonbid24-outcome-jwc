import argparse
from collections import OrderedDict
import itertools
import json
import numpy as np
from typing import Any, Callable, Dict, List

from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import lightning.pytorch as pl

class WarmupConstantLR(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_t: int=0, warmup_factor_init: float=0, last_epoch=-1):
        assert warmup_factor_init <=1 and warmup_factor_init >=0
        def lr_lambda(step):
            if step < warmup_t:
                return warmup_factor_init + (1-warmup_factor_init) * float(step) / float(max(1.0, warmup_t))
            return 1.
        
        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)