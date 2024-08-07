from __future__ import annotations

import torch


class BatchBoundingBoxes:
    """An image can have multiple boxes. This class stores bounding boxes across multiple images."""

    def __init__(self, boxes_tensor: torch.Tensor):
        """2d array of boxes is a list of bounding boxes in the format [x1, y1, x2, y2]

        Args:
            boxes_tensor (torch.Tensor): tensor of shape (b, nBB, 4)
        """
        self._bboxes = boxes_tensor

    @property
    def bboxes(self):
        return self._bboxes

    @classmethod
    def from_anchors_and_rpn_bbox_offset_volume(
        cls,
        anchor_sizes: torch.Tensor,
        anchor_ratios: torch.Tensor,
        offset_volume: torch.Tensor,
        image_size: tuple[int, int],
    ):
        """Generate bounding boxes from the anchor boxes and the bounding box offsets.

        Args:
            anchor_sizes: torch.tensor(3)
            anchor_ratios: torch.tensor(3)
            offset_volume: torch.tensor(b, s, s, 9, 4): volume coming out of the RPN
            image_size: tuple[int, int]: w, h
        """

        b = offset_volume.shape[0]
        s = offset_volume.shape[1]

        x_scaling_ratio = image_size[0] // s
        y_scaling_ratio = image_size[1] // s

        # grid containing x offsets
        anchor_widths = anchor_sizes.unsqueeze(0).expand(3, 3).reshape(-1)
        anchor_heights = anchor_sizes.unsqueeze(1).expand(3, 3).reshape(-1)
        ratios = anchor_ratios.repeat(3)
        anchor_widths = anchor_widths * ratios
        anchor_heights = anchor_heights * ratios

        anchor_widths = torch.arange(9)  # (9) #TODO
        anchor_heights = torch.arange(9)  # (9) #TODO

        # the data changes over the minus one dim. that dim doesn't change. copy along all other dims.
        # will be more memory efficient with expand
        feature_map_grid_x_offsets = torch.arange(s).view(1, 1, -1, 1).expand(b, s, s, 1).repeat(1, 1, 1, 9) * x_scaling_ratio
        feature_map_grid_y_offsets = torch.arange(s).view(1, -1, 1, 1).expand(b, s, s, 1).repeat(1, 1, 1, 9) * y_scaling_ratio

        new_bbox_volume = torch.zeros_like(offset_volume)

        new_bbox_volume[:, :, :, :, 0] = offset_volume[:, :, :, :, 0] * anchor_widths + feature_map_grid_x_offsets  # x
        new_bbox_volume[:, :, :, :, 1] = offset_volume[:, :, :, :, 1] * anchor_heights + feature_map_grid_y_offsets  # y
        new_bbox_volume[:, :, :, :, 2] = new_bbox_volume[:, :, :, :, 0] + torch.exp(offset_volume[:, :, :, :, 2]) * anchor_widths  # x2=x1 + w
        new_bbox_volume[:, :, :, :, 3] = new_bbox_volume[:, :, :, :, 1] + torch.exp(offset_volume[:, :, :, :, 3]) * anchor_heights  # y2=y1 + h

        bboxes = offset_volume.reshape(b, s * s * 9, 4)

        return cls(bboxes)

    @classmethod
    def from_bounding_boxes_and_offsets(cls, batch_bboxes: BatchBoundingBoxes, offsets: torch.Tensor) -> BatchBoundingBoxes:
        """Given bounding boxes and offsets, adjust the bounding boxes.

        Args:
            bboxes (BatchBoundingBoxes): bounding boxes
            offsets (torch.Tensor): offsets to adjust the bounding boxes with shape (b, nBB, 4)
        """

        prev_boxes = batch_bboxes.bboxes

        prev_boxes_width = prev_boxes[:, :, 2] - prev_boxes[:, :, 0]
        prev_boxes_height = prev_boxes[:, :, 3] - prev_boxes[:, :, 1]

        new_boxes = torch.zeros_like(prev_boxes)

        new_boxes[:, :, 0] = offsets[:, :, 0] * prev_boxes_width + prev_boxes[:, :, 0]  # x1
        new_boxes[:, :, 1] = offsets[:, :, 1] * prev_boxes_height + prev_boxes[:, :, 1]  # y1
        new_boxes[:, :, 2] = new_boxes[:, :, 0] + prev_boxes_width * torch.exp(offsets[:, :, 2])  # x2 = x1 + w
        new_boxes[:, :, 3] = new_boxes[:, :, 1] + prev_boxes_height * torch.exp(offsets[:, :, 3])  # y2 = y1 + h

        return cls(new_boxes)

    @staticmethod
    def pick_top_k_boxes_per_image(
        objectness: torch.Tensor, bboxes: BatchBoundingBoxes, k: int, pos_to_neg_ratio: float, pos_iou: float, neg_iou: float
    ) -> BatchBoundingBoxes:
        """Pick the top boxes per batch.

        Args:
            objectness (torch.Tensor): objectness scores of the boxes of shape (b,)
            num_bboxes (int): number of boxes to pick
            pos_to_neg_ratio (float): ratio of positive to negative boxes
        """

        # self._bboxes =
        self._bboxes.sort

        raise NotImplementedError
