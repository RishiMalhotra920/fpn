import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset

from fpn.models.fast_rcnn_classifier import FastRCNNClassifier
from fpn.models.fpn import FPN
from fpn.models.rpn import RPN
from fpn.utils.batch_bounding_boxes import BatchBoundingBoxes


class FasterRCNNWithFPN(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        nms_threshold: float,
        num_rpn_rois_to_sample: int = 2000,
        rpn_pos_to_neg_ratio: float = 0.33,
        rpn_pos_iou: float = 0.7,
        rpn_neg_iou: float = 0.3,
    ):
        super().__init__()
        self.backbone = FPN()
        self.fpn_map_small_anchor_scales = torch.tensor([32, 64, 128])
        self.fpn_map_medium_anchor_scales = torch.tensor([64, 128, 256])
        self.fpn_map_large_anchor_scales = torch.tensor([128, 256, 512])
        anchor_ratios = torch.tensor([0.5, 1, 2])
        self.all_anchor_scales = [
            self.fpn_map_small_anchor_scales,
            self.fpn_map_medium_anchor_scales,
            self.fpn_map_large_anchor_scales,
        ]
        self.all_anchor_ratios = [anchor_ratios, anchor_ratios, anchor_ratios]

        self.rpn = RPN(
            in_channels=256,
            num_anchor_scales=len(self.fpn_map_small_anchor_scales),
            num_anchor_ratios=len(anchor_ratios),
        )
        self.fast_rcnn_classifier = FastRCNNClassifier(num_classes=3)

        self.image_size = image_size
        self.nms_threshold = nms_threshold
        self.num_rpn_rois_to_sample = num_rpn_rois_to_sample
        self.rpn_pos_to_neg_ratio = rpn_pos_to_neg_ratio
        self.rpn_pos_iou = rpn_pos_iou
        self.rpn_neg_iou = rpn_neg_iou

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, BatchBoundingBoxes, torch.Tensor, BatchBoundingBoxes, torch.Tensor, BatchBoundingBoxes]:
        """Forward pass of the model.

        Passes the image through the FPN backbone, then the RPN, then the FastRCNNClassifier.

        Args:
            x (torch.Tensor): input image of shape (B, 3, H, W)

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
                all_background_objectness (list[torch.Tensor]): list of background objectness scores of shape (b, k*1/pos_to_neg_ratio)
                all_background_bboxes (list[torch.Tensor]): list of background bounding boxes of shape (b, k*1/pos_to_neg_ratio, 4)
                all_foreground_objectness (list[torch.Tensor]): list of foreground objectness scores of shape (b, k*pos_to_neg_ratio)
                all_foreground_bboxes (list[torch.Tensor]): list of foreground bounding boxes of shape (b, k*pos_to_neg_ratio, 4)
                all_fast_rcnn_cls (list[torch.Tensor]): list of fast rcnn classifier softmax scores of shape (b, nBB, num_classes)
                all_fast_rcnn_bboxes (list[torch.Tensor]): list of fast rcnn classifier bounding box regression offsets of shape (b, nBB, num_classes, 4)
        """

        fpn_maps = self.backbone(x)

        all_objectness = []
        all_bboxes = []

        for fpn_map, anchor_scale, anchor_ratio in zip(
            fpn_maps, self.all_anchor_scales, self.all_anchor_ratios
        ):  # (B, F, H/2, W/2), then (B, F, H/4, W/4), then (B, F, H/8, W/8)
            objectness, bbox_offset_volume = self.rpn(fpn_map)  # (b, s*s*9), (b, s*s*9, 4)

            bboxes = BatchBoundingBoxes.from_anchors_and_rpn_bbox_offset_volume(
                self.anchor_sizes, self.anchor_ratios, bbox_offset_volume, self.image_size
            )

            all_objectness.append(objectness)
            all_bboxes.append(bboxes.bboxes)

        # all objectness and bounding boxes from all the fpn maps
        objectness = torch.cat(all_objectness, dim=1)
        rpn_bboxes = BatchBoundingBoxes(torch.cat(all_bboxes, dim=1))

        background_objectness, background_bboxes, foreground_objectness, foreground_bboxes = (
            self.pick_foreground_and_background_objectness_and_bounding_boxes(
                objectness,
                rpn_bboxes,
                k=self.num_rpn_rois_to_sample,
                pos_to_neg_ratio=self.rpn_pos_to_neg_ratio,
                pos_iou=self.rpn_pos_iou,
                neg_iou=self.rpn_neg_iou,
            )
        )  # (b, k*1/pos_to_neg_ratio), (b, k*1/pos_to_neg_ratio, 4), (b, k*pos_to_neg_ratio), (b, k*pos_to_neg_ratio, 4)

        # pass to the fast rcnn classifier
        cls_probs, bbox_offsets_for_all_classes = self.fast_rcnn_classifier(
            fpn_maps, foreground_bboxes
        )  # (b, nBB, num_classes), (b, nBB, num_classes,4)

        # for each image, pick the class with the highest softmax score and use the corresponding bounding box
        fast_rcnn_cls, bbox_offsets = self._pick_top_class_and_bbox_offsets(cls_probs, bbox_offsets_for_all_classes)  # (b, nBB), (b, nBB, 4)

        fast_rcnn_bboxes = BatchBoundingBoxes.from_bounding_boxes_and_offsets(foreground_bboxes, bbox_offsets)

        # (b, nBB*1/pos_to_neg_ratio), (b, nBB*1/pos_to_neg_ratio, 4), (b, nBB*pos_to_neg_ratio), (b, nBB*pos_to_neg_ratio, 4), (b, nBB), (b, nBB, 4)
        return (objectness, rpn_bboxes, foreground_objectness, foreground_bboxes, fast_rcnn_cls, fast_rcnn_bboxes)

    def pick_foreground_and_background_objectness_and_bounding_boxes(
        self, objectness: torch.Tensor, batch_bounding_boxes: BatchBoundingBoxes, k: int, pos_to_neg_ratio: float, pos_iou: float, neg_iou: float
    ) -> tuple[torch.Tensor, BatchBoundingBoxes, torch.Tensor, BatchBoundingBoxes]:
        """Apply non max suprression and pick the top k foreground and background objectness scores and their corresponding bounding boxes.

        If there are not enough foreground or background boxes, we repeat sample the positive or negative boxes.

        Args:
            objectness (torch.Tensor): objectness scores of the boxes of shape (b, s*s*9)
            batch_bounding_boxes (torch.Tensor): bounding boxes in this batch of shape (b, s*s*9, 4)
            k (torch.Tensor): number of bounding boxes to select
            pos_to_neg_ratio (float): number of pos:neg select
            pos_iou (float): the iou over which we classify an object as foreground
            neg_iou (float): the iou below which we classify an object as negative
        """

        bboxes = batch_bounding_boxes.bboxes
        bboxes = torchvision.ops.nms(bboxes, objectness, self.nms_threshold)
        is_predicted_background = objectness < neg_iou  # (b, s*s*9)
        is_predicted_foreground = objectness > pos_iou  # (b, s*s*9)

        background_objectness = objectness[is_predicted_background]  # (b, k)
        foreground_objectness = objectness[is_predicted_foreground]  # (b, k)
        background_bbox = bboxes[is_predicted_background]  # (b, k, 4)
        foreground_bbox = bboxes[is_predicted_foreground]  # (b, k, 4)

        # if there are not enough foreground or background boxes, we repeat sample the positive or negative boxes using
        # the good old modulo operator. the modulo basically wraps around the index if it exceeds the length of the array.
        random_neg_index = (
            torch.randperm(int(k * (1 - pos_to_neg_ratio))) % background_objectness.shape[1]
        )  # (k*(1-pos_to_neg_ratio)) eg: [3, 1, 2, 0, ... 891]
        picked_background_objectness = background_objectness[:, random_neg_index]
        picked_background_bbox = background_bbox[:, random_neg_index]

        random_index = torch.randperm(int(k * pos_to_neg_ratio)) % foreground_objectness.shape[1]  # (k*pos_to_neg_ratio) eg: [3, 1, 2, 0, ... 891]

        picked_foreground_objectness = foreground_objectness[:, random_index]  # (b, k)
        picked_foreground_bbox = foreground_bbox[:, random_index]  # (b, k, 4)

        return (
            picked_background_objectness,
            BatchBoundingBoxes(picked_background_bbox),
            picked_foreground_objectness,
            BatchBoundingBoxes(picked_foreground_bbox),
        )

    def _pick_top_class_and_bbox_offsets(
        self, cls_probs: torch.Tensor, bbox_offsets_for_all_classes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick the class with the highest softmax score and use the corresponding bounding box for each image in the batch.

        Returns the class and bounding box offsets for each image in the batch.

        Args:
            cls_probs (torch.Tensor): probability of each class for each bounding box in the batch of shape (b, nBB, num_classes)
            bbox_offsets_for_all_classes (_type_): bounding box offsets for all classes of shape (b, nBB, num_classes, 4)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: class and bounding box offsets for each image in the batch of shape (b, nBB), (b, nBB, 4)
        """
        b = cls_probs.shape[0]
        num_boxes = cls_probs.shape[1]
        batch_idx = torch.arange(b).unsqueeze(1).expand(b, num_boxes)  # (b, nBB) [[0, 1, 2, 3],[0, 1, 2, 3],[0, 1, 2, 3]]
        box_idx = torch.arange(num_boxes).unsqueeze(0).expand(b, num_boxes)  # (b, nBB) [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
        cls = cls_probs.argmax(dim=2)  # (b, nBB) eg: [[0, 2, 1, 3], [4, 4, 2, 3], [5, 1, 2, 3]]
        bbox_offsets = bbox_offsets_for_all_classes[batch_idx, box_idx, cls]  # (b, nBB, 4)
        return cls, bbox_offsets  # (b, nBB), (b, nBB, 4)

    def predict(self, dataset: Dataset) -> list[np.ndarray]:
        """Given a dataset, predict the output of the model on the dataset.

        Args:
            dataset (Dataset): A dataset object.

        Returns:
            np.ndarray: A numpy array of predictions
                Shape: list[nx6]. ret[i] represents bounding boxes in image i.
                Shape of ret[i]: nx6 where n is the number of bounding boxes in image i
                and 6 is the number of values in the bounding box: x1, y1, x2, y2, confidence, class of the bounding
        """
        raise NotImplementedError
