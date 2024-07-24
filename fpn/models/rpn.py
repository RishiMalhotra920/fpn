import torch
import torch.nn as nn
import torch.nn.functional as F


class RPN(nn.Module):
    def __init__(self, in_channels, anchor_scales, anchor_ratios):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(512, len(anchor_scales) * len(anchor_ratios), kernel_size=1)
        self.bbox_pred = nn.Conv2d(512, len(anchor_scales) * len(anchor_ratios) * 4, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        cls = self.cls_layer(x)
        bbox = self.bbox_pred(x)
        return cls, bbox


# TODO: create_anchors to translate the anchor boxes from image space to feature map space.
