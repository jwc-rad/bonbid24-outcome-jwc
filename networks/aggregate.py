from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange

from monai.networks.layers.utils import get_act_layer, get_dropout_layer, get_norm_layer
       
class MeanPool(nn.Module):
    """
    input shape: (batch, size, channel)
    """
    def __init__(
        self,
        in_channels: int = None,
        reduce_channels: int = None,
    ) -> None:
        super().__init__()
        if reduce_channels is not None:
            assert(isinstance(in_channels, int))
            self.reduce = nn.Linear(in_channels, reduce_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'reduce'):
            x = self.reduce(x)
        x = torch.mean(x, dim=1)
        return x
    
class MeanPool2(nn.Module):
    """
    input shape: (batch, size, channel)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_first: bool = True,
    ) -> None:
        super().__init__()
        self._fc = nn.Linear(in_channels, out_channels)
        self.pool_first = pool_first
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_first: 
            x = torch.mean(x, dim=1)
        else:
            x = self._fc(x)
            x = torch.mean(x, dim=1)            
        return x
    
class MaxPool(nn.Module):
    """
    input shape: (batch, size, channel)
    """
    def __init__(
        self,
        in_channels: int = None,
        reduce_channels: int = None,
    ) -> None:
        super().__init__()
        if reduce_channels is not None:
            assert(isinstance(in_channels, int))
            self.reduce = nn.Linear(in_channels, reduce_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'reduce'):
            x = self.reduce(x)
        x, _ = torch.max(x, dim=1)
        return x
    
class Attention(nn.Module):
    """
    input shape: (batch, size, channel)
    Ref:
        monai.networks.nets.milmodel
        Attention-based Deep Multiple Instance Learning (https://arxiv.org/abs/1802.04712)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 2048,
        reduce_channels: int = None,
    ) -> None:
        super().__init__()
        x_channels = in_channels
        if reduce_channels is not None:
            self.reduce = nn.Linear(in_channels, reduce_channels)
            x_channels = reduce_channels
        self.attention = nn.Sequential(
            nn.Linear(x_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )
        
    def forward(self, x: torch.Tensor, returns=False) -> torch.Tensor:
        if hasattr(self, 'reduce'):
            x = self.reduce(x)
        a = self.attention(x)
        a = torch.softmax(a, dim=1)
        x = torch.sum(x * a, dim=1)
        if returns:
            return x, a
        return x
    
class ReduceAttention(nn.Module):
    """
    input shape: (batch, size, channel)
    Ref:
        monai.networks.nets.milmodel
        Attention-based Deep Multiple Instance Learning (https://arxiv.org/abs/1802.04712)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 2048,
        reduce_channels: int = 512,
    ) -> None:
        super().__init__()
        self.reduce = nn.Linear(in_channels, reduce_channels)
        self.attention = nn.Sequential(
            nn.Linear(reduce_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )
        self._fc = nn.Linear(reduce_channels, out_channels)
        
    def forward(self, x: torch.Tensor, return_features=False) -> torch.Tensor:
        x = self.reduce(x)
        a = self.attention(x)
        a = torch.softmax(a, dim=1)
        x = torch.sum(x * a, dim=1)
        if not return_features:
            x = self._fc(x)
        return x
    
    
class LSTMLast(nn.LSTM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, input, hx=None):
        output, hidden = super().forward(input, hx=hx)
        return output[:,-1,:]
    
class LSTMFlatten(nn.LSTM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, input, hx=None):
        output, hidden = super().forward(input, hx=hx)
        output = output.reshape(output.shape[0], -1)
        return output

class GRULast(nn.GRU):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, input, hx=None):
        output, hidden = super().forward(input, hx=hx)
        return output[:,-1,:]
    
class GRUMean(nn.GRU):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, input, hx=None):
        output, hidden = super().forward(input, hx=hx)
        return torch.mean(output, dim=1)
    
class GRUFlatten(nn.GRU):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, input, hx=None):
        output, hidden = super().forward(input, hx=hx)
        output = output.reshape(output.shape[0], -1)
        return output