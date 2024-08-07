import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

from fpn.utils.batch_bounding_boxes import BatchBoundingBoxes


class FastRCNNClassifier(nn.Module):
    """FastRCNNClassifier takes in the cropped feature map of arbitrary size, RoI pools it
    and produces softmax scores and bounding box regression offsets.

    The bounding box regression offsets are used to adjust the bounding box coordinates output by the RPN.

    the network outputs a (B, num_classes) tensor for the softmax scores and a (B, num_classes * 4) tensor for the
    bbox scores. pick the class with the highest score and use the corresponding bbox.
    """

    def __init__(self, num_classes: int, dropout_prob: float = 0.5):
        super().__init__()
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0, sampling_ratio=-1)
        # mean and variance computed across the batch and across the spatial dimensions.
        # one batch norm for each channel. reduces the # params and mean and variances are more stable
        self.bn_after_roi_pool = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(7 * 7 * 256, 1024)
        # batch norm across each neuron
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.cls = nn.Linear(1024, num_classes)
        self.bbox_layer = nn.Linear(1024, num_classes * 4)

    def forward(self, x: torch.Tensor, rois: BatchBoundingBoxes) -> tuple[torch.Tensor, torch.Tensor]:
        """Takes in a batch of feature maps and RoIs in all those feature maps and returns the softmax scores and bounding box regression offsets.

        Args:
            x (torch.Tensor): cropped feature map of arbitrary size with 256 channels
            rois (torch.Tensor): bounding box coordinates in the format (batch_index, x1, y1, x2, y2)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: softmax scores and bounding box regression offsets
        """
        assert x.shape[1] == 256, f"Expected 256 channels, got {x.shape[1]} channels"
        roi_boxes = rois.bboxes
        x = self.roi_align(x, rois)  # (b, num_rois, 256, 7, 7)
        x = self.bn_after_roi_pool(x)  # (b, num_rois, 256, 7, 7)
        x = x.view(x.size(0), -1)  # (b, num_rois, 256*7*7)
        x = F.relu(self.bn1(self.fc1(x)))  # (b, num_rois, 1024)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))  # (b, num_rois, 1024)
        x = self.dropout(x)
        cls = F.softmax(self.cls(x), dim=1)  # (b, num_rois, num_classes)
        bbox = self.bbox_layer(x)  # (b, num_rois, num_classes * 4)
        bbox = bbox.view(-1, cls.shape[1], 4)  # (b, num_rois, num_classes, 4)

        return cls, bbox  # (b, num_rois, num_classes), (b, num_rois, num_classes, 4)
