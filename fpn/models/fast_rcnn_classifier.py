import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign


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

    def __call__(self, x: torch.Tensor, rois: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return super().__call__(x, rois)

    def forward(self, x: torch.Tensor, rois: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Takes in a batch of feature maps and RoIs in all those feature maps and returns the softmax scores and bounding box regression offsets.

        Args:
            x (torch.Tensor): cropped feature map of arbitrary size with 256 channels
            rois (list[torch.Tensor]): list of RoIs in corner format. Each RoI is a tensor of shape (num_rois, 4)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: softmax scores and bounding box regression offsets
        """
        assert x.shape[1] == 256, f"Expected 256 channels, got {x.shape[1]} channels"
        # the roi align layer takes in a batch of feature maps and a list of RoIs and
        # does the RoI pooling per image correctly but returns a single tensor with all the pooled RoIs.
        num_bboxes_per_image = [roi.shape[0] for roi in rois]
        x = self.roi_align(x, rois)  # (total_num_rois, 256, 7, 7)
        x = self.bn_after_roi_pool(x)  # (total_num_rois, 256, 7, 7)
        x = x.view(x.size(0), -1)  # (total_num_rois, 256*7*7)
        x = F.relu(self.bn1(self.fc1(x)))  # (total_num_rois, 1024)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))  # (total_num_rois, 1024)
        x = self.dropout(x)
        cls = F.softmax(self.cls(x), dim=1)  # (total_num_rois, num_classes)
        bboxes = self.bbox_layer(x)  # (total_num_rois, num_classes * 4)
        bboxes = bboxes.view(-1, cls.shape[1], 4)  # (b, num_rois, num_classes, 4)
        list_of_cls = list(torch.split(cls, num_bboxes_per_image))  # tuple[(num_rois, num_classes)]
        list_of_bboxes = list(torch.split(bboxes, num_bboxes_per_image))  # tuple[(num_rois, num_classes, 4)]

        return list_of_cls, list_of_bboxes  # list[(num_rois, num_classes)], list[(num_rois, num_classes, 4)]
