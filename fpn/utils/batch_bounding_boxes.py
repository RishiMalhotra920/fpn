from __future__ import annotations

import torch


class BatchBoundingBoxes:
    """An image can have multiple boxes. This class stores bounding boxes across multiple images."""

    def __init__(self, boxes_tensor: torch.Tensor, format: str = "corner"):
        """
        Initialize BoundingBoxes with a tensor of bounding boxes.

        Args:
            boxes_tensor (torch.Tensor): tensor of shape (b, nBB, 4) where:
                If format is 'corner': [x1, y1, x2, y2]
                If format is 'center': [cx, cy, w, h]
            format (str): 'corner' for [x1, y1, x2, y2] or 'center' for [cx, cy, w, h]
        """
        if format not in ["corner", "center"]:
            raise ValueError("format must be either 'corner' or 'center'")

        if boxes_tensor.dim() != 3 or boxes_tensor.shape[-1] != 4:
            raise ValueError("boxes_tensor must have shape (b, nBB, 4)")

        if format == "center":
            self._bbox = self._center_to_corner(boxes_tensor)
        else:
            self._bbox = boxes_tensor

    @staticmethod
    def _center_to_corner(boxes):
        new_boxes = torch.empty_like(boxes)
        new_boxes[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1
        new_boxes[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1
        new_boxes[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2
        new_boxes[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2
        return new_boxes

    @staticmethod
    def _corner_to_center(boxes):
        new_boxes = torch.empty_like(boxes)
        new_boxes[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2  # cx
        new_boxes[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2  # cy
        new_boxes[..., 2] = boxes[..., 2] - boxes[..., 0]  # w
        new_boxes[..., 3] = boxes[..., 3] - boxes[..., 1]  # h
        return new_boxes

    @property
    def corner_format(self):
        return self._bbox

    @property
    def center_format(self):
        return self._corner_to_center(self._bbox)

    @classmethod
    def convert_rpn_bbox_offsets_to_rpn_bbox(
        cls,
        anchor_heights: torch.Tensor,
        anchor_widths: torch.Tensor,
        offset_volume: torch.Tensor,
        image_size: tuple[int, int],
        device: str,
    ):
        """Takes in rpn bbox offsets and anchor positions and returns the bounding boxes.

        Args:
            anchor_heights: tensor of shape (9, ) - heights of the anchor boxes
            anchor_widths: tensor of shape (9, ) - heights of the anchor boxes
            offset_volume: torch.tensor(b, s, s, 9, 4): volume coming out of the RPN
            image_size: tuple[int, int]: w, h
            b: int: batch size
            f: int: number of feature maps
            s: int: size of the feature map
        """

        b, s, _, _, _ = offset_volume.shape

        feature_map_x_step = image_size[0] / s
        feature_map_y_step = image_size[1] / s

        y_grid_cell_centers = ((torch.arange(0, s, device=device).float() * feature_map_x_step) + (feature_map_y_step / 2)).reshape(1, s, 1, 1)
        x_grid_cell_centers = ((torch.arange(0, s, device=device).float() * feature_map_x_step) + (feature_map_y_step / 2)).reshape(1, 1, s, 1)

        anchor_with_offset_positions = torch.zeros_like(offset_volume, device=device)  # (b, s, s, 3, 3, 4)
        # print('devices', anchor_with_offset_positions.device, offset_volume.device, anchor_widths.device, x_grid_cell_centers.device)

        anchor_with_offset_positions[:, :, :, :, 0] = offset_volume[:, :, :, :, 0] * anchor_widths + x_grid_cell_centers  # x = t_x * w + x
        anchor_with_offset_positions[:, :, :, :, 1] = offset_volume[:, :, :, :, 1] * anchor_heights + y_grid_cell_centers
        anchor_with_offset_positions[:, :, :, :, 2] = (
            x_grid_cell_centers + torch.exp(offset_volume[:, :, :, :, 2]) * anchor_widths
        )  # x2=x1 + exp(t_w) * w
        anchor_with_offset_positions[:, :, :, :, 3] = (
            y_grid_cell_centers + torch.exp(offset_volume[:, :, :, :, 3]) * anchor_heights
        )  # y2=y1 + exp(t_h) * h

        bounding_boxes_xywh = anchor_with_offset_positions.reshape(b, s * s * 9, 4)

        bounding_boxes_xyxy = cls(bounding_boxes_xywh, format="center")

        return bounding_boxes_xyxy

    # @classmethod
    # def from_bounding_boxes_and_offsets(cls, batch_bbox: BatchBoundingBoxes, offsets: torch.Tensor) -> BatchBoundingBoxes:
    #     """Given bounding boxes and offsets, adjust the bounding boxes.

    #     Args:
    #         bbox (BatchBoundingBoxes): bounding boxes
    #         offsets (torch.Tensor): offsets to adjust the bounding boxes with shape (b, nBB, 4)
    #     """

    #     prev_boxes = batch_bbox.corner_format

    #     prev_boxes_width = prev_boxes[:, :, 2] - prev_boxes[:, :, 0]
    #     prev_boxes_height = prev_boxes[:, :, 3] - prev_boxes[:, :, 1]

    #     new_boxes = torch.zeros_like(prev_boxes)

    #     new_boxes[:, :, 0] = offsets[:, :, 0] * prev_boxes_width + prev_boxes[:, :, 0]  # x1
    #     new_boxes[:, :, 1] = offsets[:, :, 1] * prev_boxes_height + prev_boxes[:, :, 1]  # y1
    #     new_boxes[:, :, 2] = new_boxes[:, :, 0] + prev_boxes_width * torch.exp(offsets[:, :, 2])  # x2 = x1 + w
    #     new_boxes[:, :, 3] = new_boxes[:, :, 1] + prev_boxes_height * torch.exp(offsets[:, :, 3])  # y2 = y1 + h

    #     return cls(new_boxes)
