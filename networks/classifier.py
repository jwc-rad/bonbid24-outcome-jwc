from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange

from monai.networks.layers.factories import Pool
from monai.networks.layers.utils import get_act_layer, get_dropout_layer, get_norm_layer

#from coral_pytorch.layers import CoralLayer

class MLP(nn.Sequential):
    """Modified from torchvision.ops.misc.MLP
    input shape: (batch, size, channel) if use_linear else (batch, channel, size)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[List[int]] = None,
        norm: Optional[Union[Tuple, str]] = None,
        act: Optional[Union[Tuple, str]] = None,
        bias: bool = True,
        dropout: Optional[Union[Tuple, str, float]] = None,
        spatial_dims: int = 1,
        use_linear= True,
        last_act: Optional[Union[Tuple, str]] = None,
    ) -> None:
        if hidden_channels is None:
            hidden_channels = []
        pwconv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][spatial_dims - 1]
        
        layers = []
        in_dim = in_channels
        for i, hidden_dim in enumerate(hidden_channels):
            if use_linear:
                layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            else:
                layers.append(pwconv(in_dim, hidden_dim, kernel_size=1, bias=bias))
            if norm is not None:
                layers.append(get_norm_layer(name=norm, spatial_dims=1, channels=hidden_dim))    
            if act is not None:
                layers.append(get_act_layer(act))
            if i < len(hidden_channels) - 1 and dropout is not None:
                layers.append(get_dropout_layer(name=dropout, dropout_dim=1))
            in_dim = hidden_dim

        if dropout is not None:
            layers.append(get_dropout_layer(name=dropout, dropout_dim=1))
        
        if use_linear:
            layers.append(nn.Linear(in_dim, out_channels, bias=bias))
        else:
            layers.append(pwconv(in_dim, out_channels, kernel_size=1, bias=bias))
                
        if last_act is not None:
            layers.append(get_act_layer(last_act))
                
        super().__init__(*layers)

class DualInputMLP(nn.Module):
    """
    input shape: (batch, size, channel) if use_linear else (batch, channel, size)
    concat x1, x2 at start of forward
    """
    def __init__(
        self,
        in1_channels: int,
        in2_channels: int,
        out_channels: int,
        hidden_channels: Optional[List[int]] = None,
        norm: Optional[Union[Tuple, str]] = None,
        act: Optional[Union[Tuple, str]] = None,
        bias: bool = True,
        dropout: Optional[Union[Tuple, str, float]] = None,
        spatial_dims: int = 1,
        use_linear= True,
        last_act: Optional[Union[Tuple, str]] = None,
    ) -> None:
        super().__init__()

        in_channels = in1_channels + in2_channels
        self.mlp = MLP(in_channels, out_channels, hidden_channels, norm, act, bias, dropout, spatial_dims, use_linear, last_act)
        self.use_linear = use_linear

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=-1) if self.use_linear else torch.cat([x1, x2], dim=1)
        x = self.mlp(x)
        return x

class FlatMLP(nn.Sequential):
    """Modified from torchvision.ops.misc.MLP
    input shape: (batch, size, channel) if use_linear else (batch, channel, size)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[List[int]] = None,
        norm: Optional[Union[Tuple, str]] = None,
        act: Optional[Union[Tuple, str]] = None,
        bias: bool = True,
        dropout: Optional[Union[Tuple, str, float]] = None,
        spatial_dims: int = 1,
        use_linear= True,
        last_act: Optional[Union[Tuple, str]] = None,
    ) -> None:
        if hidden_channels is None:
            hidden_channels = []
        pwconv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][spatial_dims - 1]
        
        avg_pool_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]
        
        layers = [
            avg_pool_type(1),
            nn.Flatten(1),
        ]
        in_dim = in_channels
        for i, hidden_dim in enumerate(hidden_channels):
            if use_linear:
                layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            else:
                layers.append(pwconv(in_dim, hidden_dim, kernel_size=1, bias=bias))
            if norm is not None:
                layers.append(get_norm_layer(name=norm, spatial_dims=1, channels=hidden_dim))    
            if act is not None:
                layers.append(get_act_layer(act))
            if i < len(hidden_channels) - 1 and dropout is not None:
                layers.append(get_dropout_layer(name=dropout, dropout_dim=1))
            in_dim = hidden_dim

        if dropout is not None:
            layers.append(get_dropout_layer(name=dropout, dropout_dim=1))
        
        if use_linear:
            layers.append(nn.Linear(in_dim, out_channels, bias=bias))
        else:
            layers.append(pwconv(in_dim, out_channels, kernel_size=1, bias=bias))
                
        if last_act is not None:
            layers.append(get_act_layer(last_act))
                
        super().__init__(*layers)
