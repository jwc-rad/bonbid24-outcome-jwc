import torch
from torch import nn

import timm

# backbones with reducing linear layer at top

class ConvNeXtV2_nano(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        pretrained: bool = True,
    ):
        super().__init__()

        net = timm.create_model('convnextv2_nano.fcmae_ft_in22k_in1k', pretrained=pretrained, in_chans=in_channels)

        nfc = net.head.fc.in_features
        if out_channels is None:
            net.head.fc = nn.Identity()
        else:
            net.head.fc = nn.Linear(nfc, out_channels, bias=True)
        
        self.net = net
        self.nfc = nfc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
class ConvNeXtV2_tiny(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        pretrained: bool = True,
    ):
        super().__init__()

        net = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=pretrained, in_chans=in_channels)

        nfc = net.head.fc.in_features
        if out_channels is None:
            net.head.fc = nn.Identity()
        else:
            net.head.fc = nn.Linear(nfc, out_channels, bias=True)
        
        self.net = net
        self.nfc = nfc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
class SEResNeXt50(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        pretrained: bool = True,
    ):
        super().__init__()

        net = timm.create_model('seresnext50_32x4d.gluon_in1k', pretrained=pretrained, in_chans=in_channels)

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
    
class SEResNeXt26d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        pretrained: bool = True,
    ):
        super().__init__()

        net = timm.create_model('seresnext26d_32x4d.bt_in1k', pretrained=pretrained, in_chans=in_channels)

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
    
class MobileNetV3_small(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        pretrained: bool = True,
    ):
        super().__init__()

        net = timm.create_model('mobilenetv3_small_100.lamb_in1k', pretrained=pretrained, in_chans=in_channels)

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
    
class MobileNetV3_large(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        pretrained: bool = True,
    ):
        super().__init__()

        net = timm.create_model('mobilenetv3_large_100.miil_in21k_ft_in1k', pretrained=pretrained, in_chans=in_channels)

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
    