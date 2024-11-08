import numpy as np
from typing import Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import UnetrBasicBlock, UnetOutBlock #, UnetrUpBlock
from monai.networks.layers.factories import Act, split_args
from monai.utils import ensure_tuple_rep

from mislight.networks.blocks.dynunet_block import UnetUpBlock
from mislight.networks.nets.mix_transformer import MixVisionTransformer

class MixUNETREncoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        num_stages: int = 4,
        num_layers: Union[Sequence[int], int] = [2,2,2,2],
        num_heads: Union[Sequence[int], int] = [1,2,5,8],
        patch_sizes: Union[Sequence[Sequence[int]], Sequence[int], int] = [7,3,3,3],
        strides: Union[Sequence[Sequence[int]], Sequence[int], int] = [4,2,2,2],
        paddings: Optional[Union[Sequence[int], int]] = None,
        sr_ratios: Union[Sequence[int], int] = [8,4,2,1],
        embed_dims: Union[Sequence[int], int] = 32,
        mlp_ratios: Union[Sequence[int], int] = 4,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        act: Union[tuple, str] = 'GELU',
        norm1: Union[Tuple, str] = 'layer',
        norm2: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        res_block: bool = True,
    ):    
        
        super().__init__()
        self.MiT = MixVisionTransformer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_stages=num_stages,
            num_layers=num_layers,
            num_heads=num_heads,
            patch_sizes=patch_sizes,
            strides=strides,
            paddings=paddings,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act,
            norm_layer=norm1,
            out_indices=[i for i in range(num_stages)],
        )
        
        self.patch_sizes = ensure_tuple_rep(patch_sizes, num_stages)
        self.strides = ensure_tuple_rep(strides, num_stages)
        
        num_heads = ensure_tuple_rep(num_heads, num_stages)
        embed_dims = ensure_tuple_rep(embed_dims, num_stages)
        filters = [i*j for i,j in zip(num_heads, embed_dims)]
        self.filters = filters
        
        encoders = []
        encoders.append(UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=filters[0],
            kernel_size=3,
            stride=1,
            norm_name=norm2,
            res_block=res_block,
        ))        
        for i in range(num_stages):
            encoders.append(UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=filters[i],
                out_channels=filters[i],
                kernel_size=3,
                stride=1,
                norm_name=norm2,
                res_block=res_block,
            ))
        self.encoders = nn.ModuleList(encoders)
     
    def forward(self, x):
        hidden_states_out = self.MiT(x)
        skips = [e(x1) for x1, e in zip([x] + hidden_states_out, self.encoders)]
        return skips    
    
class MixUNETREncoderv2(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        num_stages: int = 4,
        num_layers: Union[Sequence[int], int] = [2,2,2,2],
        num_heads: Union[Sequence[int], int] = [1,2,5,8],
        patch_sizes: Union[Sequence[Sequence[int]], Sequence[int], int] = [7,3,3,3],
        strides: Union[Sequence[Sequence[int]], Sequence[int], int] = [4,2,2,2],
        paddings: Optional[Union[Sequence[int], int]] = None,
        sr_ratios: Union[Sequence[int], int] = [8,4,2,1],
        embed_dims: Union[Sequence[int], int] = 32,
        mlp_ratios: Union[Sequence[int], int] = 4,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        act: Union[tuple, str] = 'GELU',
        norm: Union[Tuple, str] = 'layer',
    ):    
        
        super().__init__()
        self.MiT = MixVisionTransformer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_stages=num_stages,
            num_layers=num_layers,
            num_heads=num_heads,
            patch_sizes=patch_sizes,
            strides=strides,
            paddings=paddings,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act,
            norm_layer=norm,
            out_indices=[i for i in range(num_stages)],
        )
        
        self.patch_sizes = ensure_tuple_rep(patch_sizes, num_stages)
        self.strides = ensure_tuple_rep(strides, num_stages)
        
        num_heads = ensure_tuple_rep(num_heads, num_stages)
        embed_dims = ensure_tuple_rep(embed_dims, num_stages)
        filters = [i*j for i,j in zip(num_heads, embed_dims)]
        self.filters = filters
             
    def forward(self, x):
        hidden_states_out = self.MiT(x)
        skips = [x] + hidden_states_out
        return skips
    
class MixUNETRDecoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        out_channels: int = 2,
        num_stages: int = 4,
        kernels: Union[Sequence[Sequence[int]], Sequence[int], int] = 3,
        strides: Union[Sequence[Sequence[int]], Sequence[int], int] = 2,
        upsample_kernels: Optional[Sequence[Union[Sequence[int], int]]] = 2,
        filters: Union[Sequence[int], int] = 32,
        drop: float = 0.0,
        norm: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        head_act: Optional[Union[Tuple, str]] = None,
        res_block: bool = True,
        upsample_mode: str = "deconv",
        **upsample_kwargs,
    ):    
        super().__init__()

        if upsample_kernels is None:
            upsample_kernels = kernels
        kernels = ensure_tuple_rep(kernels, num_stages)
        strides = ensure_tuple_rep(strides, num_stages)
        upsample_kernels = ensure_tuple_rep(upsample_kernels, num_stages)
        filters = ensure_tuple_rep(filters, num_stages)
        filters = filters + (filters[-1],)
        
        decoders = []     
        for i in range(num_stages):
            decoders.append(UnetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=filters[i],
                out_channels=filters[i+1],
                kernel_size=kernels[i],
                stride=strides[i],
                upsample_kernel_size=upsample_kernels[i],
                norm_name=norm,
                res_block=res_block,
                upsample_mode=upsample_mode,
                **upsample_kwargs,
            ))
        self.decoders = nn.ModuleList(decoders)
        
        self.head = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=filters[-1], out_channels=out_channels,
        )
        
        if head_act is None:
            self.head_act = nn.Identity()
        else:
            _act, _act_args = split_args(head_act)
            self.head_act = Act[_act](**_act_args)
        
    def forward(self, skips):
        skips = skips[::-1]
        
        x = skips[0]
        
        for skip, up in zip(skips[1:], self.decoders):
            x = up(x, skip)
        
        x = self.head_act(self.head(x))
        
        return x
    
class MixUNETRDecoderv2(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 2,
        num_stages: int = 4,
        kernels: Union[Sequence[Sequence[int]], Sequence[int], int] = 3,
        strides: Union[Sequence[Sequence[int]], Sequence[int], int] = 2,
        upsample_kernels: Optional[Sequence[Union[Sequence[int], int]]] = 2,
        filters: Union[Sequence[int], int] = 32,
        drop: float = 0.0,
        norm: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        head_act: Optional[Union[Tuple, str]] = None,
        res_block: bool = True,
        upsample_mode: str = "deconv",
        **upsample_kwargs,
    ):    
        super().__init__()

        if upsample_kernels is None:
            upsample_kernels = kernels
        kernels = ensure_tuple_rep(kernels, num_stages)
        strides = ensure_tuple_rep(strides, num_stages)
        upsample_kernels = ensure_tuple_rep(upsample_kernels, num_stages)
        filters = ensure_tuple_rep(filters, num_stages)
        
        pres = []
        pres.append(UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=filters[-1],
            kernel_size=3,
            stride=1,
            norm_name=norm,
            res_block=res_block,
        ))        
        for i in range(1, num_stages+1):
            pres.append(UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=filters[-i],
                out_channels=filters[-i],
                kernel_size=3,
                stride=1,
                norm_name=norm,
                res_block=res_block,
            ))
        self.pres = nn.ModuleList(pres)
        
        filters = filters + (filters[-1],)
        ups = []     
        for i in range(num_stages):
            ups.append(UnetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=filters[i],
                out_channels=filters[i+1],
                kernel_size=kernels[i],
                stride=strides[i],
                upsample_kernel_size=upsample_kernels[i],
                norm_name=norm,
                res_block=res_block,
                upsample_mode=upsample_mode,
                **upsample_kwargs,
            ))
        self.ups = nn.ModuleList(ups)
        
        self.head = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=filters[-1], out_channels=out_channels,
        )
        
        if head_act is None:
            self.head_act = nn.Identity()
        else:
            _act, _act_args = split_args(head_act)
            self.head_act = Act[_act](**_act_args)
        
    def forward(self, skips):
        skips = [pre(x) for pre, x in zip(self.pres, skips)]
        skips = skips[::-1]
        
        x = skips[0]
        for skip, up in zip(skips[1:], self.ups):
            x = up(x, skip)
        
        x = self.head_act(self.head(x))
        
        return x
        
class MixUNETR(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 2,
        num_stages: int = 4,
        num_layers: Union[Sequence[int], int] = [2,2,2,2],
        num_heads: Union[Sequence[int], int] = [1,2,5,8],
        patch_sizes: Union[Sequence[Sequence[int]], Sequence[int], int] = 3,
        strides: Union[Sequence[Sequence[int]], Sequence[int], int] = 2,
        upsample_kernels: Optional[Sequence[Union[Sequence[int], int]]] = 2,
        paddings: Optional[Union[Sequence[int], int]] = None,
        sr_ratios: Union[Sequence[int], int] = [8,4,2,1],
        embed_dims: Union[Sequence[int], int] = 32,
        mlp_ratios: Union[Sequence[int], int] = 4,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        act: Union[tuple, str] = 'GELU',
        norm1: Union[Tuple, str] = 'layer',
        norm2: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        head_act: Optional[Union[Tuple, str]] = None,
        res_block: bool = True,
        upsample_mode: str = "deconv",
        **upsample_kwargs,
    ):    
        super().__init__()
        
        self.encoder = MixUNETREncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_stages=num_stages,
            num_layers=num_layers,
            num_heads=num_heads,
            patch_sizes=patch_sizes,
            strides=strides,
            paddings=paddings,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act=act,
            norm1=norm1,
            norm2=norm2,
            res_block=res_block,
        )
        
        filters = self.encoder.filters[::-1]
        kernels = ensure_tuple_rep(patch_sizes, num_stages)[::-1]
        strides = ensure_tuple_rep(strides, num_stages)[::-1]
        upsample_kernels = ensure_tuple_rep(upsample_kernels, num_stages)[::-1]
        self.decoder = MixUNETRDecoder(
            spatial_dims=spatial_dims,
            out_channels=out_channels,
            num_stages=num_stages,
            kernels=kernels,
            strides=strides,
            upsample_kernels=upsample_kernels,
            filters=filters,
            drop=drop,
            norm=norm2,
            head_act=head_act,
            res_block=res_block,
            upsample_mode=upsample_mode,
            **upsample_kwargs,
        )
        
    def forward(self, x):
        skips = self.encoder(x)
        x = self.decoder(skips)
        return x  
    
    
class MixUNETRv2(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 2,
        num_stages: int = 4,
        num_layers: Union[Sequence[int], int] = [2,2,2,2],
        num_heads: Union[Sequence[int], int] = [1,2,5,8],
        patch_sizes: Union[Sequence[Sequence[int]], Sequence[int], int] = 3,
        strides: Union[Sequence[Sequence[int]], Sequence[int], int] = 2,
        upsample_kernels: Optional[Sequence[Union[Sequence[int], int]]] = 2,
        paddings: Optional[Union[Sequence[int], int]] = None,
        sr_ratios: Union[Sequence[int], int] = [8,4,2,1],
        embed_dims: Union[Sequence[int], int] = 32,
        mlp_ratios: Union[Sequence[int], int] = 4,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        act: Union[tuple, str] = 'GELU',
        norm1: Union[Tuple, str] = 'layer',
        norm2: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        head_act: Optional[Union[Tuple, str]] = None,
        res_block: bool = True,
        upsample_mode: str = "deconv",
        **upsample_kwargs,
    ):    
        super().__init__()
        
        self.encoder = MixUNETREncoderv2(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_stages=num_stages,
            num_layers=num_layers,
            num_heads=num_heads,
            patch_sizes=patch_sizes,
            strides=strides,
            paddings=paddings,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act=act,
            norm=norm1,
        )
        
        filters = self.encoder.filters[::-1]
        kernels = ensure_tuple_rep(patch_sizes, num_stages)[::-1]
        strides = ensure_tuple_rep(strides, num_stages)[::-1]
        upsample_kernels = ensure_tuple_rep(upsample_kernels, num_stages)[::-1]
        
        self.decoder = MixUNETRDecoderv2(
            spatial_dims=spatial_dims,
            out_channels=out_channels,
            num_stages=num_stages,
            kernels=kernels,
            strides=strides,
            upsample_kernels=upsample_kernels,
            filters=filters,
            drop=drop,
            norm=norm2,
            head_act=head_act,
            res_block=res_block,
            upsample_mode=upsample_mode,
            **upsample_kwargs,
        )
        
    def forward(self, x):
        skips = self.encoder(x)
        x = self.decoder(skips)
        return x  