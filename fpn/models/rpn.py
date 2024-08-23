import torch
import torch.nn as nn
import torch.nn.functional as F


class RPN(nn.Module):
    """
    Region Proposal Network (RPN) with batch normalization.

    This network slides over the feature map tensor to produce objectness scores
    and bounding box regression offsets for anchor boxes.
    """

    def __init__(self, in_channels: int, num_anchor_scales: int, num_anchor_ratios: int, device: str):
        super().__init__()
        self.num_anchor_scales = num_anchor_scales
        self.num_anchor_ratios = num_anchor_ratios
        self.device = device

        # Convolutional layer with batch normalization
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        # Classification (objectness) layer
        self.cls_conv = nn.Conv2d(512, num_anchor_scales * num_anchor_ratios, kernel_size=1)
        self.cls_bn = nn.BatchNorm2d(num_anchor_scales * num_anchor_ratios)

        # Bounding box regression layer
        self.bbox_conv = nn.Conv2d(512, num_anchor_scales * num_anchor_ratios * 4, kernel_size=1)
        self.bbox_bn = nn.BatchNorm2d(num_anchor_scales * num_anchor_ratios * 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the RPN.

        Args:
            x (torch.Tensor): Input tensor of shape (b, f1, h, w)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - rpn_objectness_pred: Objectness scores of shape (b, s, s, 9)
                - rpn_bbox_offset_volume_pred: Bounding box offsets of shape (b, s, s, 9, 4)
        """
        b, _, s, _ = x.shape

        # Apply conv1 with batch norm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))  # (b, 512, s, s)

        # Objectness prediction
        rpn_objectness_pred = self.cls_conv(x)  # (b, 9, s, s)
        rpn_objectness_pred = self.cls_bn(rpn_objectness_pred)
        rpn_objectness_pred = F.sigmoid(rpn_objectness_pred)
        rpn_objectness_pred = rpn_objectness_pred.permute(0, 2, 3, 1)  # (b, s, s, 9)

        # Bounding box offset prediction
        rpn_bbox_offset_volume_pred = self.bbox_conv(x)  # (b, 36, s, s)
        rpn_bbox_offset_volume_pred = self.bbox_bn(rpn_bbox_offset_volume_pred)
        rpn_bbox_offset_volume_pred = rpn_bbox_offset_volume_pred.permute(0, 2, 3, 1)  # (b, s, s, 36)
        rpn_bbox_offset_volume_pred = rpn_bbox_offset_volume_pred.view(b, s, s, 9, 4)  # (b, s, s, 9, 4)

        return rpn_objectness_pred, rpn_bbox_offset_volume_pred
