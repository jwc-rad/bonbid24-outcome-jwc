from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss



class MSEActLoss(_Loss):
    def __init__(
        self,
        sigmoid: bool = False,
        softmax: bool = False,
        reduction: str = 'mean',
    ) -> None:
        super().__init__(reduction=reduction)
        if int(sigmoid) + int(softmax) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True].")
        self.sigmoid = sigmoid
        self.softmax = softmax
        
    def forward(self, input: Tuple[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)
            #target = torch.sigmoid(target)
        
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1) 
                #target = torch.softmax(target, 1) 
        
        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
            
        return F.mse_loss(input, target, reduction=self.reduction)
    
class MAEActLoss(_Loss):
    def __init__(
        self,
        sigmoid: bool = False,
        softmax: bool = False,
        reduction: str = 'mean',
    ) -> None:
        super().__init__(reduction=reduction)
        if int(sigmoid) + int(softmax) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True].")
        self.sigmoid = sigmoid
        self.softmax = softmax
        
    def forward(self, input: Tuple[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)
            #target = torch.sigmoid(target)
        
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1) 
                #target = torch.softmax(target, 1) 
        
        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
            
        return F.l1_loss(input, target, reduction=self.reduction)
    

class MSEVarLoss(_Loss):
    def __init__(
        self,
        var_weight: float = 1,
        reduction: str = 'mean',
    ) -> None:
        super().__init__(reduction=reduction)
        self.var_weight = var_weight
        
    def forward(self, input: Tuple[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """
        input: (mean, var)
        """
        mean = input[0]
        var = input[1] * self.var_weight
        f = 0.5 * (torch.mul(torch.exp(-var), F.mse_loss(mean, target, reduction='none')) + var)
        
        if self.reduction == "mean":
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == "sum":
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == "none":
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return f.float()
    
class MAEVarLoss(_Loss):
    def __init__(
        self,
        var_weight: float = 1,
        reduction: str = 'mean',
    ) -> None:
        super().__init__(reduction=reduction)
        self.var_weight = var_weight
        
    def forward(self, input: Tuple[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """
        input: (mean, var)
        """
        mean = input[0]
        var = input[1] * self.var_weight
        f = 0.5 * (torch.mul(torch.exp(-var), F.l1_loss(mean, target, reduction='none')) + var)
        
        if self.reduction == "mean":
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == "sum":
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == "none":
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return f
    
class BCEWithLogitsConLoss(_Loss):
    """
    torch.nn.BCEWithLogitsLoss with targets also as logits
    """
    def __init__(
        self,
        temperature_y: float = 1,
        reduction: str = 'mean',
    ) -> None:
        """
        Args:
            temperature_y: soften(t > 1) or sharpen(t < 1) predictions.
        """
        super().__init__(reduction=reduction)
        self.temperature_y = temperature_y
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = input
        y = target / self.temperature_y
        y = torch.sigmoid(y)

        loss = F.binary_cross_entropy_with_logits(x, y, reduction=self.reduction)
        return loss
        
        
    
class KLDivLoss(_Loss):
    """
    torch.nn.KLDivLoss with inputs/target as just logits or softmax (not log_softmax)
    """
    def __init__(
        self,
        reduction: str = 'batchmean',
        softmax = True,
        softmax_target = False,
        clamp_eps: float = None,
        temperature: float = 1,
    ) -> None:
        """
        Args:
            softmax: if True, apply a softmax function to the prediction
        """
        super().__init__(reduction=reduction)
        self.softmax = softmax
        self.softmax_target = softmax_target
        self.clamp_eps = clamp_eps
        assert temperature > 0
        if temperature != 1:
            assert softmax and softmax_target
        
        self.temperature = temperature
            
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = input / self.temperature
        y = target / self.temperature
        if self.softmax:
            x = torch.softmax(x, 1)
        if self.softmax_target:
            y = torch.softmax(y, 1)
        if isinstance(self.clamp_eps, float):
            x = torch.clamp(x, self.clamp_eps)
            y = torch.clamp(y, self.clamp_eps)    
        x = x.log()
        loss = F.kl_div(x, y, reduction=self.reduction, log_target=False)
        return loss * (self.temperature ** 2)
    
class JSDivLoss(_Loss):
    """
    Jensen Shannon Divergence
    input/target as just logits or softmax (not log_softmax)
    """
    def __init__(
        self,
        reduction: str = 'batchmean',
        softmax = True,
        clamp_eps: float = None,
    ) -> None:
        """
        Args:
            softmax: if True, apply a softmax function to the prediction
        """
        super().__init__(reduction=reduction)
        self.softmax = softmax
        self.clamp_eps = clamp_eps
            
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = input
        y = target
        if self.softmax:
            x = torch.softmax(x, 1)
            y = torch.softmax(y, 1)
        if isinstance(self.clamp_eps, float):
            x = torch.clamp(x, self.clamp_eps)  
            y = torch.clamp(y, self.clamp_eps)    
        x = x.log()
        y = y.log()
        kl_xy = F.kl_div(x, y, reduction=self.reduction, log_target=True)
        kl_yx = F.kl_div(y, x, reduction=self.reduction, log_target=True)
        
        return 0.5*(kl_xy + kl_yx)
    
    
class CosineDistanceLoss(_Loss):
    """
    loss decrease = similarity increase, distance decrease
    """
    def __init__(
        self,
        margin: float = 0.,
        reduction: str = 'mean',
    ) -> None:
        super().__init__(reduction=reduction)
        self.margin = margin
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = torch.ones(input.size(0), device=input.device)
        return F.cosine_embedding_loss(input, target, y, margin=self.margin, reduction=self.reduction)

class CosineSimilarityLoss(_Loss):
    """
    loss decrease = similarity decrease, distance increase
    """
    def __init__(
        self,
        margin: float = -1.,
        reduction: str = 'mean',
    ) -> None:
        super().__init__(reduction=reduction)
        self.margin = margin
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = -torch.ones(input.size(0), device=input.device)
        return F.cosine_embedding_loss(input, target, y, margin=self.margin, reduction=self.reduction)


class PairwiseDistanceLoss(_Loss):
    def __init__(
        self,
        p: float = 2.,
        eps: float = 1e-6,
        reduction: str = 'mean',
    ) -> None:
        super().__init__(reduction=reduction)
        
        self.dist = torch.nn.PairwiseDistance(p=p, eps=eps)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        f = self.dist(input, target)
                
        if self.reduction == "mean":
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == "sum":
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == "none":
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return f
    
class NegativePairwiseDistanceLoss(_Loss):
    def __init__(
        self,
        p: float = 2.,
        eps: float = 1e-6,
        reduction: str = 'mean',
    ) -> None:
        super().__init__(reduction=reduction)
        
        self.dist = torch.nn.PairwiseDistance(p=p, eps=eps)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        f = -self.dist(input, target)
                
        if self.reduction == "mean":
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == "sum":
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == "none":
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return f
    
class NormalizedPairwiseDistanceLoss(_Loss):
    def __init__(
        self,
        p: float = 2.,
        eps: float = 1e-6,
        reduction: str = 'mean',
    ) -> None:
        super().__init__(reduction=reduction)
        
        self.p = p
        self.eps = eps
        self.dist = torch.nn.PairwiseDistance(p=p, eps=eps)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        u = F.normalize(input, p=self.p, eps=self.eps)
        v = F.normalize(target, p=self.p, eps=self.eps)
        f = self.dist(u, v)
                
        if self.reduction == "mean":
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == "sum":
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == "none":
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return f
    
class NegativeNormalizedPairwiseDistanceLoss(_Loss):
    def __init__(
        self,
        p: float = 2.,
        eps: float = 1e-6,
        reduction: str = 'mean',
    ) -> None:
        super().__init__(reduction=reduction)
        
        self.p = p
        self.eps = eps
        self.dist = torch.nn.PairwiseDistance(p=p, eps=eps)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        u = F.normalize(input, p=self.p, eps=self.eps)
        v = F.normalize(target, p=self.p, eps=self.eps)
        f = -self.dist(u, v)
                
        if self.reduction == "mean":
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == "sum":
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == "none":
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return f