import torch
from torch import nn

import torchvision.models as tvm
from monai.networks.nets.efficientnet import EfficientNetBN

# backbones with reducing linear layer at top

class ResNet18Backbone(nn.Module):
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

class ResNet50Backbone(nn.Module):
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
    
class ResNeXt50_32X4DBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'ResNeXt50_32X4D_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.resnext50_32x4d(weights=weights)
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

class DenseNet121Backbone(nn.Module):
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
    
class DenseNet201Backbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'DenseNet201_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.densenet201(weights=weights)
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
    
class MobileNet_V3_SmallBackbone(nn.Module):
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

class MobileNet_V3_LargeBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'MobileNet_V3_Large_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.mobilenet_v3_large(weights=weights)
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

class ShuffleNet_V2_X0_5Backbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'ShuffleNet_V2_X0_5_Weights.DEFAULT',
        out_channels: int = None,
        **kwargs,
    ):
        super().__init__()
        
        net = tvm.shufflenet_v2_x0_5(weights=weights, **kwargs,)
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
    
class ShuffleNet_V2_X1_0Backbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'ShuffleNet_V2_X1_0_Weights.DEFAULT',
        out_channels: int = None,
        **kwargs,
    ):
        super().__init__()
        
        net = tvm.shufflenet_v2_x1_0(weights=weights, **kwargs,)
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
    
class SqueezeNet1_0Backbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'SqueezeNet1_0_Weights.DEFAULT',
        out_channels: int = None,
        
    ):
        super().__init__()
        
        net = tvm.squeezenet1_0(weights=weights)
        if in_channels != 3:
            net.features[0] = nn.Conv2d(in_channels, 96, kernel_size=7, stride=2)
        nfc = net.classifier[1].in_channels
        if out_channels is None:
            net.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        else:
            net.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(nfc, out_channels, bias=True))
        
        self.net = net
        self.nfc = nfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x

class SqueezeNet1_1Backbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'SqueezeNet1_1_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.squeezenet1_1(weights=weights)
        if in_channels != 3:
            net.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2)
        nfc = net.classifier[1].in_channels
        if out_channels is None:
            net.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        else:
            net.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(nfc, out_channels, bias=True))
        
        self.net = net
        self.nfc = nfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
class MNASNet0_5Backbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'MNASNet0_5_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.mnasnet0_5(weights=weights)
        if in_channels != 3:
            net.layers[0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
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


class RegNet_Y_1_6GFBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'RegNet_Y_1_6GF_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.regnet_y_1_6gf(weights=weights)
        if in_channels != 3:
            net.stem[0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
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

class EfficientNet_B0Backbone(nn.Module):
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
    
class EfficientNet_B1Backbone(nn.Module):
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
    
class EfficientNet_B2Backbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'EfficientNet_B2_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.efficientnet_b2(weights=weights)
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
    
class EfficientNet_B3Backbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'EfficientNet_B3_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.efficientnet_b3(weights=weights)
        if in_channels != 3:
            net.features[0][0] = nn.Conv2d(in_channels, 40, kernel_size=3, stride=2, padding=1, bias=False)
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

class EfficientNet_B4Backbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'EfficientNet_B4_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.efficientnet_b4(weights=weights)
        if in_channels != 3:
            net.features[0][0] = nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1, bias=False)
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
    
class EfficientNet_V2_SBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'EfficientNet_V2_S_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.efficientnet_v2_s(weights=weights)
        if in_channels != 3:
            net.features[0][0] = nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
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
    
class ConvNeXt_SmallBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'ConvNeXt_Small_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.convnext_small(weights=weights)
        if in_channels != 3:
            net.features[0][0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)
        nfc = net.classifier[2].in_features
        if out_channels is None:
            net.classifier = nn.Flatten(1)
        else:
            net.classifier = nn.Sequential(nn.Flatten(1), nn.Linear(nfc, out_channels, bias=True))
        
        self.net = net
        self.nfc = nfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
class ConvNeXt_TinyBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'ConvNeXt_Tiny_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.convnext_tiny(weights=weights)
        if in_channels != 3:
            net.features[0][0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)
        nfc = net.classifier[2].in_features
        if out_channels is None:
            net.classifier = nn.Flatten(1)
        else:
            net.classifier = nn.Sequential(nn.Flatten(1), nn.Linear(nfc, out_channels, bias=True))
        
        self.net = net
        self.nfc = nfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
class Swin_TBackbone(nn.Module):
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
    
class Swin_V2_TBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'Swin_V2_T_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.swin_v2_t(weights=weights)
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
    
class ViT_B_32Backbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        weights = 'ViT_B_32_Weights.DEFAULT',
        out_channels: int = None,
    ):
        super().__init__()
        
        net = tvm.vit_b_32(weights=weights)
        if in_channels != 3:
            net.conv_proj = nn.Conv2d(in_channels, 768, kernel_size=32, stride=32)
        nfc = net.heads.head.in_features
        if out_channels is None:
            net.heads.head = nn.Identity()
        else:
            net.heads.head = nn.Linear(nfc, out_channels, bias=True)
        
        self.net = net
        self.nfc = nfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x