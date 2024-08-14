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

    def __init__(self, in_channels, num_anchor_scales, num_anchor_ratios):
        super().__init__()
        self.num_anchor_scales = num_anchor_scales
        self.num_anchor_ratios = num_anchor_ratios
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(512, num_anchor_scales * num_anchor_ratios, kernel_size=1)
        self.bbox_layer = nn.Conv2d(512, num_anchor_scales * num_anchor_ratios * 4, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """forward pass on rpn

        Args:
            x (torch.Tensor): (b, f1, h, w)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: cls, bbox of shapes (b, s*s*9), (b, f2, s, s, 9, 4)
        """
        b = x.shape[0]
        x = F.relu(self.conv1(x))
        cls = self.cls_layer(x)
        bbox = self.bbox_layer(x).reshape(b, -1, 4)
        print("shapes", cls.shape, bbox.shape)
        return cls, bbox


# TODO: create_anchors to translate the anchor boxes from image space to feature map space.
