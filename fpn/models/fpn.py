import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = list(resnet.children())[:-2]
        self.initial_layers = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = self.backbone[4]  # (N, 256, H, W)
        self.layer2 = self.backbone[5]  # (N, 512, H, W)
        self.layer3 = self.backbone[6]  # (N, 1024, H/2, W/2)
        self.layer4 = self.backbone[7]  # (N, 2048, H/4, W/4)

        self.lateral4 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.lateral3 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.lateral2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.lateral1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.initial_layers(x)  # (N, 64, M, M)
        c1 = self.layer1(x)  # (N, 256, M/2, M)
        c2 = self.layer2(c1)  # (N, 512, M/4, M/2)
        c3 = self.layer3(c2)  # (N, 1024, M/8, M/4)
        c4 = self.layer4(c3)  # (N, 2048, M/16, M/8)

        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        p1 = self.lateral1(c1) + F.interpolate(p2, scale_factor=2, mode="nearest")

        # smooth to remove aliasing effects.
        p4_smooth = self.smooth4(p4)  # (N, 256, 56, 56)
        p3_smooth = self.smooth3(p3)  # (N, 256, 28, 28)
        p2_smooth = self.smooth2(p2)  # (N, 256, 14, 14)
        p1_smooth = self.smooth1(p1)  # (N, 256, 7, 7)

        return p1_smooth, p2_smooth, p3_smooth, p4_smooth
