import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset

from fpn.data.VOC_data import BACKGROUND_CLASS_INDEX
from fpn.models.fast_rcnn_classifier import FastRCNNClassifier
from fpn.models.rpn import RPN
from fpn.utils.batch_bounding_boxes import BatchBoundingBoxes


class FasterRCNN(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        nms_threshold: float,
        *,
        device: str,
        fast_rcnn_dropout: float,
        num_rpn_rois_to_sample: int,
        rpn_pos_to_neg_ratio: float,
        rpn_pos_iou: float,
        rpn_neg_iou: float,
        rpn_pred_to_gt_match_iou_threshold: float = 0.5,
    ):
        super().__init__()
        self.image_size = image_size
        self.nms_threshold = nms_threshold
        self.num_rpn_rois_to_sample = num_rpn_rois_to_sample
        self.rpn_pos_to_neg_ratio = rpn_pos_to_neg_ratio
        self.rpn_pos_iou = rpn_pos_iou
        self.rpn_neg_iou = rpn_neg_iou
        self.rpn_pred_to_gt_match_iou_threshold = rpn_pred_to_gt_match_iou_threshold

        # these changes 2
        self.rpn = RPN(in_channels=256, num_anchor_scales=3, num_anchor_ratios=3, device=device).to(device)
        self.fast_rcnn_classifier = FastRCNNClassifier(num_classes=21, dropout=fast_rcnn_dropout).to(device)
        self.device = device

    def __call__(
        self,
        fpn_map: torch.Tensor,
        anchor_heights: torch.Tensor,
        anchor_widths: torch.Tensor,
        anchor_positions: torch.Tensor,
        raw_cls_gt: torch.Tensor,
        raw_bbox_gt: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        # list[torch.Tensor],
        # list[torch.Tensor],
        # list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        # list[torch.Tensor],
        # list[torch.Tensor],
        # float,
        # float,
    ]:
        return super().__call__(fpn_map, anchor_heights, anchor_widths, anchor_positions, raw_cls_gt, raw_bbox_gt)

    def forward(
        self,
        fpn_map: torch.Tensor,
        anchor_heights: torch.Tensor,
        anchor_widths: torch.Tensor,
        anchor_positions: torch.Tensor,
        raw_cls_gt: torch.Tensor,
        raw_bbox_gt: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        # list[torch.Tensor],
        # list[torch.Tensor],
        # list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        # list[torch.Tensor],
        # list[torch.Tensor],
        # float,
        # float,
    ]:
        """Forward pass of the model.

        Passes the image through the FPN backbone, then the RPN, then the FastRCNNClassifier.

        Args:
            x (torch.Tensor): input image of shape (B, 3, H, W)

        Returns:
        """

        b = fpn_map.shape[0]
        b, _, s, _ = fpn_map.shape

        # n_a is number of anchors
        rpn_objectness_pred, rpn_bbox_offset_pred = self.rpn(fpn_map)  # (b, s, s, n_a), (b, s, s, n_a, 4)

        rpn_bbox_pred = BatchBoundingBoxes.convert_rpn_bbox_offsets_to_rpn_bbox(
            anchor_heights, anchor_widths, rpn_bbox_offset_pred, self.image_size, self.device
        ).corner_format  # (b, s*s*n_a, 4)

        rpn_bbox_offset_pred = rpn_bbox_offset_pred.reshape(b, -1, 4)  # (b, s*s*n_a, 4)
        rpn_objectness_pred = rpn_objectness_pred.reshape(b, -1)  # (b, n_a*s*s)

        # should do this matching with anchors btw for stability.
        # get rpn_bbox_gt from here as well.
        rpn_objectness_gt, rpn_cls_gt, rpn_bbox_gt, rpn_bbox_offset_gt, rpn_bbox_pred_and_best_rpn_bbox_gt_iou = self.extract_rpn_gt(
            anchor_positions, raw_cls_gt, raw_bbox_gt, self.rpn_pred_to_gt_match_iou_threshold
        )

        # # for inference, pick the top N bboxes according to confidence scores.
        # (
        #     rpn_bbox_pred_nms_fg_and_bg_some,
        #     fast_rcnn_cls_gt_nms_fg_and_bg_some,
        #     fast_rcnn_bbox_gt_nms_fg_and_bg_some,
        #     rpn_num_fg_bbox_picked,
        #     rpn_num_bg_bbox_picked,
        # ) = self.pick_fg_and_bg_objectness_and_bbox(
        #     rpn_objectness_pred,
        #     rpn_bbox_pred,
        #     rpn_cls_gt,
        #     rpn_bbox_gt,
        #     rpn_bbox_pred_and_best_rpn_bbox_gt_iou,
        #     k=self.num_rpn_rois_to_sample,
        #     pos_to_neg_ratio=self.rpn_pos_to_neg_ratio,
        #     pos_iou=self.rpn_pos_iou,
        #     neg_iou=self.rpn_neg_iou,
        # )

        # fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox, fast_rcnn_bbox_offsets_for_all_classes_for_some_rpn_bbox = self.fast_rcnn_classifier(
        #     fpn_map, rpn_bbox_pred_nms_fg_and_bg_some
        # )  # list[(num_rois, num_classes)], list[(num_rois, num_classes, 4)]

        # fast_rcnn_bbox_offsets_pred = self.get_fast_rcnn_bbox_offsets_for_gt_class(
        #     fast_rcnn_cls_gt_nms_fg_and_bg_some, fast_rcnn_bbox_offsets_for_all_classes_for_some_rpn_bbox
        # )

        # fast_rcnn_bbox_offsets_gt = self.get_fast_rcnn_bbox_offsets_gt(rpn_bbox_pred_nms_fg_and_bg_some, fast_rcnn_bbox_gt_nms_fg_and_bg_some)

        # # fast_rcnn_bbox_pred_for_some_rpn_bbox = self.apply_offsets_to_fast_rcnn_bbox(
        # # rpn_bbox_pred_nms_fg_and_bg_some, fast_rcnn_bbox_offsets_for_gt_class_for_some_rpn_bbox
        # # )

        return (
            rpn_objectness_pred,
            rpn_bbox_offset_pred,
            # rpn_bbox_pred_nms_fg_and_bg_some,
            # fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox,
            # fast_rcnn_bbox_offsets_pred,
            rpn_objectness_gt,
            rpn_bbox_offset_gt,
            # fast_rcnn_cls_gt_nms_fg_and_bg_some,
            # fast_rcnn_bbox_offsets_gt,
            # rpn_num_fg_bbox_picked,
            # rpn_num_bg_bbox_picked,
        )

    def get_fast_rcnn_bbox_offsets_gt(
        self, rpn_bbox_pred_nms_fg_and_bg_some: list[torch.Tensor], fast_rcnn_bbox_gt_nms_fg_and_bg_some: list[torch.Tensor]
    ):
        fast_rcnn_bbox_offsets_gt = []
        for rpn_image_bbox_pred_nms_fg_and_bg_some, fast_rcnn_image_bbox_gt_nms_fg_and_bg_some in zip(
            rpn_bbox_pred_nms_fg_and_bg_some, fast_rcnn_bbox_gt_nms_fg_and_bg_some
        ):
            base_xywh = BatchBoundingBoxes._corner_to_center(rpn_image_bbox_pred_nms_fg_and_bg_some)
            base_with_offsets_xywh = BatchBoundingBoxes._corner_to_center(fast_rcnn_image_bbox_gt_nms_fg_and_bg_some)

            fast_rcnn_image_bbox_offsets_gt = torch.zeros_like(base_with_offsets_xywh, device=self.device)

            fast_rcnn_image_bbox_offsets_gt[:, 0] = (base_with_offsets_xywh[:, 0] - base_xywh[:, 0]) / base_xywh[:, 2]
            fast_rcnn_image_bbox_offsets_gt[:, 1] = (base_with_offsets_xywh[:, 1] - base_xywh[:, 1]) / base_xywh[:, 3]
            fast_rcnn_image_bbox_offsets_gt[:, 2] = torch.log(base_with_offsets_xywh[:, 2] / base_xywh[:, 2])
            fast_rcnn_image_bbox_offsets_gt[:, 3] = torch.log(base_with_offsets_xywh[:, 3] / base_xywh[:, 3])

            fast_rcnn_bbox_offsets_gt.append(fast_rcnn_image_bbox_offsets_gt)

        return fast_rcnn_bbox_offsets_gt

    def get_fast_rcnn_bbox_offsets_for_gt_class(
        self, fast_rcnn_cls_gt_nms_fg_and_bg_some: list[torch.Tensor], fast_rcnn_bbox_offsets_for_all_classes: list[torch.Tensor]
    ):
        fast_rcnn_bbox_offsets_for_gt_class = []
        for fast_rcnn_image_bbox_offsets_for_all_classes, fast_rcnn_image_cls_gt_nms_fg_and_bg_some in zip(
            fast_rcnn_bbox_offsets_for_all_classes, fast_rcnn_cls_gt_nms_fg_and_bg_some
        ):
            # fast_rcnn_image_bbox_offsets_for_all_classes: (L_i, num_classes, 4)
            # fast_rcnn_image_cls_gt_nms_fg_and_bg_some: (L_i)
            row_indices = torch.arange(fast_rcnn_image_cls_gt_nms_fg_and_bg_some.shape[0], device=self.device)
            fast_rcnn_image_bbox_offsets_for_gt_class = fast_rcnn_image_bbox_offsets_for_all_classes[
                row_indices, fast_rcnn_image_cls_gt_nms_fg_and_bg_some, :
            ]
            fast_rcnn_bbox_offsets_for_gt_class.append(fast_rcnn_image_bbox_offsets_for_gt_class)
        return fast_rcnn_bbox_offsets_for_gt_class

    def extract_rpn_gt(
        self,
        anchor_positions: torch.Tensor,
        raw_cls_gt: torch.Tensor,
        raw_bbox_gt: torch.Tensor,
        iou_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match the pred bbox to the gt bbox based on the IOU and return if the pred bbox are foreground or not.

        Here's what's happening:
            1. For each image, there are n predictions and m ground truth bbox.
            2. We match each prediction to one ground truth bbox (repeats allowed). This gives us n matches per image.
            3. However, when matching, the IoU may be pretty low, in this case we set is_foreground to False.
            4. So we end up with a matches array and a is_foreground_array for each image.

        the dataloader pads the gt_bbox with 0s. however, it doesn't matter that the 0s are included because ious will be 0.

        Args:
            pred_bbox: Union[list[torch.Tensor]:
                the rpn_preds are as a torch.Tensor of (b, z_i, 4) but
            gt_bbox (torch.Tensor): gt bbox of shape (b, max_gt_bbox, 4)
            iou_threshold (float): threshold to consider a match

        Returns:
            torch.Tensor: matched indices of shape (b, nBB)
        """
        # TODO: ensure the only for loop should be in this function.
        rpn_cls_gt = []
        rpn_bbox_gt = []
        rpn_objectness_gt = []
        rpn_bbox_offset_gt = []
        rpn_bbox_pred_and_best_rpn_bbox_gt_iou = []
        b = len(raw_bbox_gt)

        for image_idx in range(b):
            # image_bbox_pred = rpn_bbox_offset_pred[image_idx]  # (z_i, 4)
            image_all_bbox_gt = raw_bbox_gt[image_idx]  # (max_gt_bbox, 4)

            ious = torchvision.ops.box_iou(anchor_positions, image_all_bbox_gt)  # (s*s*9, max_gt_bbox)

            best_iou, best_gt_index = torch.max(ious, dim=1)  # (s*s*9)
            is_foreground = best_iou > iou_threshold  # (s*s*9)

            image_cls_gt = torch.where(
                is_foreground, raw_cls_gt[image_idx][best_gt_index], torch.full_like(best_gt_index, BACKGROUND_CLASS_INDEX, device=self.device)
            )
            image_bbox_gt = image_all_bbox_gt[best_gt_index]  # (s*s*9, 4)
            # need to convert this to offsets
            image_bbox_offsets_gt = self._convert_bbox_and_anchor_to_bbox_offsets(anchor_positions, image_bbox_gt)  # (z_i, 4)

            rpn_objectness_gt.append(is_foreground.float())
            rpn_cls_gt.append(image_cls_gt)
            rpn_bbox_gt.append(image_bbox_gt)
            rpn_bbox_pred_and_best_rpn_bbox_gt_iou.append(best_iou)  # (z_i)
            rpn_bbox_offset_gt.append(image_bbox_offsets_gt)

        return (
            torch.stack(rpn_objectness_gt),
            torch.stack(rpn_cls_gt),
            torch.stack(rpn_bbox_gt),
            torch.stack(rpn_bbox_offset_gt),
            torch.stack(rpn_bbox_pred_and_best_rpn_bbox_gt_iou),
        )

    def _convert_bbox_and_anchor_to_bbox_offsets(self, anchor_positions: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        # given anchor positions corresponding to boxes, convert the bbox to offsets
        # anchor_positions: (s*s*9, 4), bbox: (s*s*9, 4)

        anchor_xywh_positions = BatchBoundingBoxes._corner_to_center(anchor_positions)  # (s*s*9, 4)
        bbox_xywh_positions = BatchBoundingBoxes._corner_to_center(bbox)  # (s*s*9, 4)

        offsets = torch.zeros_like(anchor_xywh_positions, device=self.device)

        # x - x_a / w_a
        offsets[:, 0] = (bbox_xywh_positions[:, 0] - anchor_xywh_positions[:, 0]) / anchor_xywh_positions[:, 2]
        offsets[:, 1] = (bbox_xywh_positions[:, 1] - anchor_xywh_positions[:, 1]) / (anchor_xywh_positions[:, 3])
        offsets[:, 2] = torch.log(bbox_xywh_positions[:, 2] / anchor_xywh_positions[:, 2])
        offsets[:, 3] = torch.log(bbox_xywh_positions[:, 3] / anchor_xywh_positions[:, 3])

        return offsets

    def apply_offsets_to_fast_rcnn_bbox(self, list_of_picked_bbox: list[torch.Tensor], offsets: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply the bounding box regression offsets to the bounding boxes.

        Args:
            list_of_picked_bbox (list[b, torch.Tensor(L_i, 4)]): list of bounding boxes in corner format of shape
            offsets (list[b, torch.Tensor(L_i, 4)]): list of bounding box offsets

        Returns:
            list[b, torch.Tensor(L_i, 4)]: bounding boxes in corner format
        """

        list_of_bbox_with_offsets = []
        for image_bbox, image_bbox_offsets in zip(list_of_picked_bbox, offsets):
            image_bbox_with_offsets = torch.zeros_like(image_bbox, device=self.device)

            prev_bbox_width = image_bbox[:, 2] - image_bbox[:, 0]
            prev_bbox_height = image_bbox[:, 3] - image_bbox[:, 1]

            image_bbox_with_offsets[:, 0] = image_bbox_offsets[:, 0] * prev_bbox_width + image_bbox[:, 0]
            image_bbox_with_offsets[:, 1] = image_bbox_offsets[:, 1] * prev_bbox_height + image_bbox[:, 1]
            image_bbox_with_offsets[:, 2] = image_bbox_with_offsets[:, 0] + torch.exp(image_bbox_offsets[:, 2]) * prev_bbox_width
            image_bbox_with_offsets[:, 3] = image_bbox_with_offsets[:, 1] + torch.exp(image_bbox_offsets[:, 3]) * prev_bbox_height

            list_of_bbox_with_offsets.append(image_bbox_with_offsets)

        return list(list_of_bbox_with_offsets)

    def pick_fg_and_bg_objectness_and_bbox(
        self,
        rpn_objectness_pred: torch.Tensor,
        rpn_bbox_pred: torch.Tensor,
        rpn_cls_gt: torch.Tensor,
        rpn_bbox_gt: torch.Tensor,
        rpn_bbox_pred_and_best_rpn_bbox_gt_iou: torch.Tensor,
        *,
        k: int,
        pos_to_neg_ratio: float,
        pos_iou: float,
        neg_iou: float,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], float, float]:
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

        (b, _) = rpn_objectness_pred.shape

        list_of_rpn_bbox_pred_nms_fg_and_bg_some = []
        list_of_fast_rcnn_cls_gt_nms_fg_and_bg_some = []
        list_of_fast_rcnn_bbox_gt_nms_fg_and_bg_some = []

        total_rpn_num_fg_bbox_picked = 0
        total_rpn_num_bg_bbox_picked = 0

        for image_idx in range(rpn_bbox_pred.shape[0]):
            # rpn_bbox_pred_nms_indices - these are the indices of the boxes that are kept after nms
            rpn_bbox_pred_nms_indices = torchvision.ops.nms(rpn_bbox_pred[image_idx], rpn_objectness_pred[image_idx], self.nms_threshold)
            rpn_bbox_pred_nms = rpn_bbox_pred[image_idx][rpn_bbox_pred_nms_indices]  # (#nms_boxes, 4)
            rpn_bbox_pred_and_best_rpn_bbox_gt_iou_nms = rpn_bbox_pred_and_best_rpn_bbox_gt_iou[image_idx, rpn_bbox_pred_nms_indices]  # (#nms_boxes)
            raw_fast_rcnn_cls_gt_nms = rpn_cls_gt[image_idx, rpn_bbox_pred_nms_indices]  # (#nms_boxes)
            raw_fast_rcnn_bbox_gt_nms = rpn_bbox_gt[image_idx, rpn_bbox_pred_nms_indices]  # (#nms_boxes, 4)

            is_rpn_pred_bg = rpn_bbox_pred_and_best_rpn_bbox_gt_iou_nms < neg_iou
            is_rpn_pred_fg = rpn_bbox_pred_and_best_rpn_bbox_gt_iou_nms > pos_iou
            # is_rpn_pred_fg = rpn_objectness_pred_nms > pos_iou  # (#foreground predicted boxes)

            total_num_pos_box = int(is_rpn_pred_fg.sum().item())
            total_num_bg_bbox = int((is_rpn_pred_bg).sum().item())
            num_fg_box_to_pick = int(pos_to_neg_ratio / (1 + pos_to_neg_ratio)) * k
            num_bg_box_to_pick = int(1 / (1 + pos_to_neg_ratio) * k)
            num_fg_box_to_pick_capped = min(total_num_pos_box, num_fg_box_to_pick)
            num_bg_box_to_pick_capped = min(total_num_bg_bbox, num_bg_box_to_pick)
            # pick some random bg bboxes

            random_neg_index = torch.randperm(num_bg_box_to_pick_capped, device=self.device)  # eg: [3, 1, 2, 0, ... 891]
            rpn_bbox_pred_nms_bg_some = rpn_bbox_pred_nms[is_rpn_pred_bg][random_neg_index, :]  # (b, some, 4)
            raw_fast_rcnn_cls_gt_nms_bg_some = raw_fast_rcnn_cls_gt_nms[is_rpn_pred_bg][random_neg_index]  # (b, some)
            raw_fast_rcnn_bbox_gt_nms_bg_some = raw_fast_rcnn_bbox_gt_nms[is_rpn_pred_bg][random_neg_index, :]  # (b, some, 4)

            # pick some random fg bboxes
            random_pos_index = torch.randperm(num_fg_box_to_pick_capped, device=self.device)  # (some) eg: [3, 1, 2, 0, ... 891]
            rpn_bbox_pred_nms_fg_some = rpn_bbox_pred_nms[is_rpn_pred_fg][random_pos_index, :]  # (b, some, 4)
            raw_fast_rcnn_cls_gt_nms_fg_some = raw_fast_rcnn_cls_gt_nms[is_rpn_pred_fg][random_pos_index]  # (b, some)
            raw_fast_rcnn_bbox_gt_nms_fg_some = raw_fast_rcnn_bbox_gt_nms[is_rpn_pred_fg][random_pos_index, :]  # (b, some, 4)

            rpn_bbox_pred_nms_fg_and_bg_some = torch.cat([rpn_bbox_pred_nms_fg_some, rpn_bbox_pred_nms_bg_some], dim=0)  # (b, k, 4)
            raw_fast_rcnn_cls_gt_nms_fg_and_bg_some = torch.cat([raw_fast_rcnn_cls_gt_nms_fg_some, raw_fast_rcnn_cls_gt_nms_bg_some], dim=0)  # (b, k)
            raw_fast_rcnn_bbox_gt_nms_fg_and_bg_some = torch.cat(
                [raw_fast_rcnn_bbox_gt_nms_fg_some, raw_fast_rcnn_bbox_gt_nms_bg_some], dim=0
            )  # (b, k, 4)
            list_of_rpn_bbox_pred_nms_fg_and_bg_some.append(rpn_bbox_pred_nms_fg_and_bg_some)
            list_of_fast_rcnn_cls_gt_nms_fg_and_bg_some.append(raw_fast_rcnn_cls_gt_nms_fg_and_bg_some)
            list_of_fast_rcnn_bbox_gt_nms_fg_and_bg_some.append(raw_fast_rcnn_bbox_gt_nms_fg_and_bg_some)

            total_rpn_num_fg_bbox_picked += num_fg_box_to_pick_capped
            total_rpn_num_bg_bbox_picked += num_bg_box_to_pick_capped

        return (
            list_of_rpn_bbox_pred_nms_fg_and_bg_some,
            list_of_fast_rcnn_cls_gt_nms_fg_and_bg_some,
            list_of_fast_rcnn_bbox_gt_nms_fg_and_bg_some,
            total_rpn_num_fg_bbox_picked / b,
            total_rpn_num_bg_bbox_picked / b,
        )

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
