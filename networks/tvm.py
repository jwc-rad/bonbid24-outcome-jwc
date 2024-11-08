import torch
from torch import nn

import torchvision.models as tvm

# backbones with reducing linear layer at top

class ResNet18(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'ResNet18_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.resnet18(weights=weights)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nfc = net.fc.in_features
        if out_channels is None:
            net.fc = nn.Identity()
        else:
            net.fc = nn.Linear(nfc, out_channels, bias=True)
        
        self.net = net
        self.nfc = nfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x

class ResNet50(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'ResNet50_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.resnet50(weights=weights)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nfc = net.fc.in_features
        if out_channels is None:
            net.fc = nn.Identity()
        else:
            net.fc = nn.Linear(nfc, out_channels, bias=True)
        
        self.net = net
        self.nfc = nfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
class DenseNet121(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'DenseNet121_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.densenet121(weights=weights)
        if in_channels != 3:
            net.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nfc = net.classifier.in_features
        if out_channels is None:
            net.classifier = nn.Identity()
        else:
            net.classifier = nn.Linear(nfc, out_channels, bias=True)
        
        self.net = net
        self.nfc = nfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
class EfficientNet_B0(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'EfficientNet_B0_Weights.DEFAULT',
        out_channels: int = None,
        **kwargs,
    ):
        super().__init__()
        
        net = tvm.efficientnet_b0(weights=weights, **kwargs,)
        if in_channels != 3:
            net.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        nfc = net.classifier[1].in_features
        if out_channels is None:
            net.classifier = nn.Identity()
        else:
            net.classifier = nn.Linear(nfc, out_channels, bias=True)
        
        self.net = net
        self.nfc = nfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x

class EfficientNet_B1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'EfficientNet_B1_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.efficientnet_b1(weights=weights)
        if in_channels != 3:
            net.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        nfc = net.classifier[1].in_features
        if out_channels is None:
            net.classifier = nn.Identity()
        else:
            net.classifier = nn.Linear(nfc, out_channels, bias=True)
        
        self.net = net
        self.nfc = nfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
class MobileNet_V3_Small(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'MobileNet_V3_Small_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.mobilenet_v3_small(weights=weights)
        if in_channels != 3:
            net.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        nfc = net.classifier[0].in_features
        if out_channels is None:
            net.classifier = nn.Identity()
        else:
            net.classifier = nn.Linear(nfc, out_channels, bias=True)
        
        self.net = net
        self.nfc = nfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x    

    
class Swin_T(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'Swin_T_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.swin_t(weights=weights)
        if in_channels != 3:
            net.features[0][0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)
        nfc = net.head.in_features
        if out_channels is None:
            net.head = nn.Identity()
        else:
            net.head = nn.Linear(nfc, out_channels, bias=True)
        
        self.net = net
        self.nfc = nfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
class ShuffleNet_V2_X1_5(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'ShuffleNet_V2_X1_5_Weights.DEFAULT',
        out_channels: int = None,
        **kwargs,
    ):
        super().__init__()
        
        net = tvm.shufflenet_v2_x1_5(weights=weights, **kwargs,)
        if in_channels != 3:
            net.conv1[0] = nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        nfc = net.fc.in_features
        if out_channels is None:
            net.fc = nn.Identity()
        else:
            net.fc = nn.Linear(nfc, out_channels, bias=True)
        
        self.net = net
        self.nfc = nfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
class ShuffleNet_V2_X2_0(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'ShuffleNet_V2_X2_0_Weights.DEFAULT',
        out_channels: int = None,
        **kwargs,
    ):
        super().__init__()
        
        net = tvm.shufflenet_v2_x2_0(weights=weights, **kwargs,)
        if in_channels != 3:
            net.conv1[0] = nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        nfc = net.fc.in_features
        if out_channels is None:
            net.fc = nn.Identity()
        else:
            net.fc = nn.Linear(nfc, out_channels, bias=True)
        
        self.net = net
        self.nfc = nfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x