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
            offset_volume: torch.tensor(b, s*s*9, 4): volume coming out of the RPN
            image_size: tuple[int, int]: w, h
        """

        # b = offset_volume.shape[0]
        d2 = offset_volume.shape[1]
        s = (d2 // 9) ** 0.5

        feature_map_x_step = image_size[0] // s
        feature_map_y_step = image_size[1] // s

        # grid containing x offsets
        anchors = torch.cartesian_prod(anchor_sizes, anchor_sizes)
        anchor_widths_unique = anchors[:, 0] * anchor_ratios  # (9, )
        anchor_heights_unique = anchors[:, 1] * (1 / anchor_ratios)  # (9, )

        feature_map_grid_x_offsets_unique = torch.arange(0, image_size[0], feature_map_x_step)
        feature_map_grid_y_offsets_unique = torch.arange(0, image_size[1], feature_map_y_step)

        # create a cartesian product of all the unique values.
        # we do this because there are s*s*9 anchor boxes at s*s locations.
        # so to translate these offsets, we need to create a cartesian product of all the unique values and say
        # - bbox # 34 is at x_offset x, y_offset y, anchor_width w, anchor_height h
        cartesian_product = torch.cartesian_prod(
            feature_map_grid_x_offsets_unique, feature_map_grid_y_offsets_unique, anchor_widths_unique, anchor_heights_unique
        )

        feature_map_grid_x_offsets = cartesian_product[:, 0]  # (s*s*9,)
        feature_map_grid_y_offsets = cartesian_product[:, 1]  # (s*s*9,)
        anchor_widths = cartesian_product[:, 2]  # (s*s*9,)
        anchor_heights = cartesian_product[:, 3]  # (s*s*9,)

        new_bbox_volume = torch.zeros_like(offset_volume)  # (b, s*s*9, 4)

        new_bbox_volume[:, :, 0] = offset_volume[:, :, 0] * anchor_widths + feature_map_grid_x_offsets  # x
        new_bbox_volume[:, :, 1] = offset_volume[:, :, 1] * anchor_heights + feature_map_grid_y_offsets  # y
        new_bbox_volume[:, :, 2] = new_bbox_volume[:, :, 0] + torch.exp(offset_volume[:, :, 2]) * anchor_widths  # x2=x1 + w
        new_bbox_volume[:, :, 3] = new_bbox_volume[:, :, 1] + torch.exp(offset_volume[:, :, 3]) * anchor_heights  # y2=y1 + h

        return cls(new_bbox_volume)

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
