from typing import List, Optional, Sequence, Tuple, Type, Union

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetOutBlock, UnetResBlock #, UnetUpBlock
from monai.networks.layers.factories import Act, split_args, Pool
from mislight.networks.blocks.dynunet_block import UnetUpBlock
from mislight.networks.nets.dynunet import DynUNetEncoder, DynUNetDecoder, DynUNet

class DynUNetAux(nn.Module):
    def __init__(
        self,
        spatial_dims: int, 
        in_channels: int,  
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Optional[Sequence[Union[Sequence[int], int]]] = None,
        filters: Union[Sequence[int], int] = 64,
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        upsample_mode: str = "deconv",
        num_blocks: Union[Sequence[int], int] = 1,
        res_block: Union[Sequence[bool], bool] = False,
        max_filters: int = 512,
        head_act: Optional[Union[Tuple, str]] = None,
        aux_channels: Optional[int] = None,
        **upsample_kwargs,
    ):    
        super().__init__()
        self.encoder = DynUNetEncoder(
            spatial_dims, in_channels, kernel_size, strides, filters, 
            dropout, norm_name, act_name, num_blocks, res_block, max_filters,
        )
        if upsample_kernel_size is None:
            upsample_kernel_size = kernel_size[1:][::-1]
        self.decoder = DynUNetDecoder(
            spatial_dims, out_channels, kernel_size[1:][::-1], strides[1:][::-1], self.encoder.filters[::-1], upsample_kernel_size[::-1],
            dropout, norm_name, act_name, upsample_mode, head_act, **upsample_kwargs,
        )
        
        avg_pool_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]
        list_auxclf = [
            avg_pool_type(1),
            nn.Flatten(1),
        ]
        if isinstance(aux_channels, int):
            list_auxclf.append(nn.Linear(self.encoder.filters[-1], aux_channels))
        self.auxclf = nn.Sequential(*list_auxclf)        
        
    def forward(self, x, return_aux=False):
        if return_aux:
            skips, feats = self.encoder(x, layers=[len(self.encoder.downsamples)])
            out = self.decoder(skips)
            out_feat = self.auxclf(feats[0])
            return out, out_feat
        else:
            skips = self.encoder(x)
            out = self.decoder(skips)
            return out