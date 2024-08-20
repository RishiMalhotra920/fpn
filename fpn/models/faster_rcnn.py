import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset

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
        num_rpn_rois_to_sample: int = 2000,
        rpn_pos_to_neg_ratio: float = 0.33,
        rpn_pos_iou: float = 0.7,
        rpn_neg_iou: float = 0.3,
        match_iou_threshold: float = 0.5,
    ):
        super().__init__()
        self.image_size = image_size
        self.nms_threshold = nms_threshold
        self.num_rpn_rois_to_sample = num_rpn_rois_to_sample
        self.rpn_pos_to_neg_ratio = rpn_pos_to_neg_ratio
        self.rpn_pos_iou = rpn_pos_iou
        self.rpn_neg_iou = rpn_neg_iou
        self.match_iou_threshold = match_iou_threshold

        # these changes 2
        self.rpn = RPN(in_channels=256, num_anchor_scales=3, num_anchor_ratios=3, device=device)
        self.fast_rcnn_classifier = FastRCNNClassifier(num_classes=21)
        self.device = device

    def __call__(
        self,
        fpn_map: torch.Tensor,
        anchor_heights: torch.Tensor,
        anchor_widths: torch.Tensor,
        gt_bbox: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        return super().__call__(fpn_map, anchor_heights, anchor_widths, gt_bbox)

    def forward(
        self,
        fpn_map: torch.Tensor,
        anchor_heights: torch.Tensor,
        anchor_widths: torch.Tensor,
        raw_bbox_gt: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
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
        rpn_objectness_pred = rpn_objectness_pred.reshape(b, -1)  # (b, n_a*s*s)

        # should do this matching with anchors btw for stability.
        rpn_objectness_gt, rpn_gt_index_match_for_rpn_bbox_pred = self.extract_rpn_objectness_and_rpn_gt_index_match_for_bbox_pred(
            rpn_bbox_pred, raw_bbox_gt, self.match_iou_threshold
        )

        # TODO: note that rpn_objectness_gt and rpn_gt_index_match_for_bbox_pred and rpn_bbox_gt
        # can all be made into one big tensor very easily. see TODO optim1
        rpn_bbox_gt = [raw_bbox_gt[i, rpn_gt_index_match_for_bbox_pred[i]] for i in range(len(rpn_objectness_gt))]

        # TODO: difference between the latter two variables???
        list_of_picked_bbox, list_of_picked_bbox_gt_matches, is_fast_rcnn_pred_foreground = self.pick_fg_and_bg_objectness_and_bbox(
            rpn_objectness_pred,
            rpn_bbox_pred,
            # rpn_objectness_gt,
            rpn_gt_index_match_for_rpn_bbox_pred,
            k=self.num_rpn_rois_to_sample,
            pos_to_neg_ratio=self.rpn_pos_to_neg_ratio,
            pos_iou=self.rpn_pos_iou,
            neg_iou=self.rpn_neg_iou,
        )

        # pass to the fast rcnn classifier
        cls_probs, bbox_offsets_for_all_classes = self.fast_rcnn_classifier(fpn_map, list_of_picked_bbox)  # list[torch.Tensor()], list[torch.Tensor]

        # # for each bbox, pick the class with the highest softmax score and use the corresponding bounding box
        # fast_rcnn_cls, bbox_offsets = self._pick_top_class_and_bbox_offsets(
        #     cls_probs, bbox_offsets_for_all_classes
        # )  # list[b, torch.Tensor(L_i, num_classes)], list[b, torch.Tensor(L_i, num_classes, 4)]

        # fast_rcnn_bbox = BatchBoundingBoxes.from_bounding_boxes_and_offsets(list_of_picked_bbox, bbox_offsets)
        # list_of_bbox_with_offsets = self.apply_offsets_to_fast_rcnn_bbox(list_of_picked_bbox, bbox_offsets)

        # torch.Tensor(b, 3*s*s*9), BatchBoundingBoxes(b, 3*s*s*9, 4), list[torch.Tensor(b, L_i)], list[torch.Tensor(b, L_i, 4)] where L_i is the number of picked bounding boxes in image i
        return (
            rpn_objectness,
            rpn_bbox.corner_format,
            is_rpn_pred_foreground,
            rpn_bbox_matches,
            cls_probs,
            bbox_offsets_for_all_classes,
            list_of_picked_bbox_gt_matches,
            is_fast_rcnn_pred_foreground,
        )

    def extract_rpn_objectness_and_rpn_gt_index_match_for_rpn_bbox_pred(
        self, rpn_bbox_pred: torch.Tensor, raw_bbox_gt: torch.Tensor, iou_threshold: float
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
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

        rpn_objectness_gt = []
        rpn_gt_index_match_for_rpn_bbox_pred = []
        b = len(rpn_bbox_pred)

        for image_idx in range(b):
            image_bbox_pred = rpn_bbox_pred[image_idx]  # (z_i, 4)
            image_bbox_gt = raw_bbox_gt[image_idx]  # (max_gt_bbox, 4)

            ious = torchvision.ops.box_iou(image_bbox_pred, image_bbox_gt)  # (z_i, max_gt_bbox)

            best_iou, best_gt_index = torch.max(ious, dim=1)  # (z_i)
            is_foreground = best_iou > iou_threshold  # (z_i)

            rpn_objectness_gt.append(is_foreground)
            rpn_gt_index_match_for_rpn_bbox_pred.append(best_gt_index)

            # assert image.shape[0] == gt_bbox

        # best_iou looks like
        # [[0.1, 0.2, 0.3, 0.4],
        #  [0.5, 0.6, 0.7, 0.8]]
        # best_gt_index looks like
        # [[0, 1, 2, 3],
        #  [0, 1, 2, 3]]
        # is_foreground looks like
        # [[False, False, False, True],
        #  [True, True, True, True]]
        # TODO optim1: make this one big tensor by torch.tensor(...)
        return rpn_objectness_gt, rpn_gt_index_match_for_rpn_bbox_pred

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
        rpn_gt_index_match_for_rpn_bbox_pred: list[torch.Tensor],
        *,
        k: int,
        pos_to_neg_ratio: float,
        pos_iou: float,
        neg_iou: float,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
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
        list_of_rpn_gt_index_match_for_rpn_bbox_pred_nms_fg_and_bg_some = []
        is_fast_rcnn_pred_foreground = []

        for image_idx in range(rpn_bbox_pred.shape[0]):
            # rpn_bbox_pred_nms_indices - these are the indices of the boxes that are kept after nms
            rpn_bbox_pred_nms_indices = torchvision.ops.nms(rpn_bbox_pred[image_idx], rpn_objectness_pred[image_idx], self.nms_threshold)
            rpn_gt_index_match_for_rpn_bbox_pred_nms = rpn_gt_index_match_for_rpn_bbox_pred[image_idx][rpn_bbox_pred_nms_indices]  # (#nms_boxes)
            rpn_bbox_pred_nms = rpn_bbox_pred[image_idx][rpn_bbox_pred_nms_indices]  # (#nms_boxes, 4)
            rpn_objectness_pred_nms = rpn_objectness_pred[image_idx][rpn_bbox_pred_nms_indices]  # (#nms_boxes)

            # is_predicted_background = rpn_objectness_pred_nms < neg_iou  # (#background predicted boxes)
            is_rpn_pred_fg = rpn_objectness_pred_nms > pos_iou  # (#foreground predicted boxes)

            # select background
            rpn_bbox_pred_nms_bg = rpn_bbox_pred_nms[~is_rpn_pred_fg]  # (#background predicted boxes, 4)
            rpn_gt_index_match_for_rpn_bbox_pred_nms_bg = rpn_gt_index_match_for_rpn_bbox_pred_nms[~is_rpn_pred_fg]  # (#background predicted boxes)
            rpn_bbox_pred_nms_fg = rpn_bbox_pred_nms[is_rpn_pred_fg]  # (#foreground predicted boxes, 4)
            rpn_gt_index_match_for_rpn_bbox_pred_nms_fg = rpn_gt_index_match_for_rpn_bbox_pred_nms[is_rpn_pred_fg]  # (#foreground predicted boxes)

            # select some of rpn_bbox_pred_nms_bg based on the ratios
            num_bg_bbox = rpn_bbox_pred_nms_bg.shape[0]
            random_neg_index = torch.randperm(int(min(num_bg_bbox, k) * (1 - pos_to_neg_ratio)), device=self.device)  # eg: [3, 1, 2, 0, ... 891]
            rpn_bbox_pred_nms_bg_some = rpn_bbox_pred_nms_bg[random_neg_index, :]  # (b, some, 4)
            rpn_gt_index_match_for_rpn_bbox_pred_nms_bg_some = rpn_gt_index_match_for_rpn_bbox_pred_nms_bg[random_neg_index]  # (b, some)

            # select some of rpn_bbox_pred_nms_bg based on the ratios
            num_fg_bbox = rpn_bbox_pred_nms_fg.shape[0]
            random_pos_index = torch.randperm(
                int(min(num_fg_bbox, k) * pos_to_neg_ratio), device=self.device
            )  # (k*pos_to_neg_ratio) eg: [3, 1, 2, 0, ... 891]
            rpn_bbox_pred_nms_fg_some = rpn_bbox_pred_nms_fg[random_pos_index, :]  # (b, min(num_foreground_bbox, k) * pos_to_neg_ratio, 4)
            rpn_gt_index_match_for_rpn_bbox_pred_nms_fg_some = rpn_gt_index_match_for_rpn_bbox_pred_nms_fg[
                random_pos_index
            ]  # (b, min(num_foreground_bbox, k) * pos_to_neg_ratio)

            rpn_bbox_pred_nms_fg_and_bg_some = torch.cat([rpn_bbox_pred_nms_fg_some, rpn_bbox_pred_nms_bg_some], dim=0)  # (b, k, 4)
            rpn_gt_index_match_for_rpn_bbox_pred_nms_fg_and_bg_some = torch.cat(
                [rpn_gt_index_match_for_rpn_bbox_pred_nms_fg_some, rpn_gt_index_match_for_rpn_bbox_pred_nms_bg_some], dim=0
            )  # (b, k)
            list_of_rpn_bbox_pred_nms_fg_and_bg_some.append(rpn_bbox_pred_nms_fg_and_bg_some)
            list_of_rpn_gt_index_match_for_rpn_bbox_pred_nms_fg_and_bg_some.append(rpn_gt_index_match_for_rpn_bbox_pred_nms_fg_and_bg_some)

            # TODO: instead of calculating is_
            is_foreground = torch.cat(
                [torch.ones_like(random_pos_index, device=self.device), torch.zeros_like(random_neg_index, device=self.device)], dim=0
            )

            is_fast_rcnn_pred_foreground.append(is_foreground)

        return list_of_rpn_bbox_pred_nms_fg_and_bg_some, list_of_rpn_gt_index_match_for_rpn_bbox_pred_nms_fg_and_bg_some, is_fast_rcnn_pred_foreground

    def _pick_top_class_and_bbox_offsets(
        self, cls_probs: list[torch.Tensor], bbox_offsets_for_all_classes: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Each image has many bbox. For each bbox, pick the class with the highest softmax score and use the corresponding bbox.

        run this method during inference! but not during training. we dont need to argmax when computing class probs.

        cls_probs[i] and bbox_offsets_for_all_classes[i] are bbox for image i

        Args:
            cls_probs (list[torch.Tensor]): probability of each class for each bounding box in the batch with list[(L_i, num_classes)]
            bbox_offsets_for_all_classes (list[torch.Tensor]): bounding box offsets for all classes of shape list[torch.tensor(L_i, num_classes, 4)]
                where L_i is the number of bounding boxes in image i.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: class and bounding box offsets for each image in the batch of shape (b, nBB), (b, nBB, 4)
        """

        all_cls = []
        all_bbox_offsets = []
        for image_cls_probs, image_bbox_offsets_for_all_classes in zip(cls_probs, bbox_offsets_for_all_classes):
            # image_cls_probs: (L_i, num_classes) where L_i is the number of bounding boxes in the image
            # image_bbox_offsets_for_all_classes: (L_i, num_classes, 4)
            num_boxes = image_cls_probs.shape[0]
            box_idx = torch.arange(num_boxes, device=self.device)
            cls = image_cls_probs.argmax(dim=1)  # (L_i)
            bbox_offsets = image_bbox_offsets_for_all_classes[box_idx, cls]  # (L_i, 4)
            assert bbox_offsets.shape == (num_boxes, 4), f"Expected (L_i, 4), got {bbox_offsets.shape}"
            all_cls.append(cls)
            all_bbox_offsets.append(bbox_offsets)

        return all_cls, all_bbox_offsets  # list[torch.Tensor(L_i)], list[torch.Tensor(L_i, 4)]

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
