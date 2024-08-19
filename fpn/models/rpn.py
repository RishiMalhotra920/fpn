import torch
import torch.nn as nn
import torch.nn.functional as F


class RPN(nn.Module):
    """returns cls: objectness scores and bbox: bounding box regression offsets.

    The RPN network needs to slide over the feature map tensor. So when we pass in the whole feature map tensor,
    this class actually slides the RPN. We then use a convolution layer to get the cls. We use another convolution layer
    to output the bounding box regression offsets. This is interesting because we usually use a fully connected layer.
    However, here we use a convolution layer to mimic the sliding window of the RPN.

    There are anchor_scales*anchor_ratios anchor boxes. For each anchor box, the network
    produces an objectness score and 4 bounding box regression offsets that modify the objectness score.

    """

    def __init__(self, in_channels: torch.Tensor, num_anchor_scales: torch.Tensor, num_anchor_ratios: torch.Tensor, device: str):
        super().__init__()
        self.num_anchor_scales = num_anchor_scales
        self.num_anchor_ratios = num_anchor_ratios
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(512, num_anchor_scales * num_anchor_ratios, kernel_size=1)
        self.bbox_layer = nn.Conv2d(512, num_anchor_scales * num_anchor_ratios * 4, kernel_size=1)
        self.device = device

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """forward pass on rpn

        Args:
            x (torch.Tensor): (b, f1, h, w)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: cls, bbox of shapes (b, s, s, 9), (b, s, s, 9, 4)
        """
        b, _, s, _ = x.shape
        b = x.shape[0]
        s = x.shape[2]
        x = F.relu(self.conv1(x))  # (b, 512, s, s)
        cls = self.cls_layer(x).reshape(b, s, s, 9)  # (b, 9, s, s) -> (b, s, s, 9)
        bbox = self.bbox_layer(x).reshape(b, s, s, 9, 4)  # (b, 9, s, s, 4) -> (b, s, s, 9, 4)
        return cls, bbox
