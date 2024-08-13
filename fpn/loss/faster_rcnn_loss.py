import torch
import torch.nn.functional as F
import torchvision
from torch import nn


class RPNLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        objectness_pred: torch.Tensor,
        bbox_pred: torch.Tensor,
        objectness_gt: torch.Tensor,
        bbox_gt: torch.Tensor,
        lambda_rpn_objectness=1,
        lambda_rpn_bbox=10,
    ) -> dict[str, torch.Tensor]:
        """Compute the RPN loss.

        objectness_pred are all the objectness predictions and objectness_gt are all the ground truth objectness matched to the anchors.

        Before computing the bbox loss, we filter out the background anchor bboxes. We only compute the bbox loss for the foreground anchor bboxes.
        bbox_pred are all the bbox predictions. bbox_gt are all the ground truth bboxes matched to the anchors.

        Args:
            objectness_pred (torch.Tensor): RPN objectness pred of shape (b, s*s*9)
            bbox_pred (torch.Tensor): RPN bbox pred of shape (b, s*s*9, 4)
            objectness_label (torch.Tensor): RPN objectness label of shape (b, s*s*9)
            bbox_label (torch.Tensor): RPN bbox label of shape (b, s*s*9, 4)
            lambda_ (int, optional): Param weighting cls_loss and bbox_loss. Defaults to 10.

        Returns:
            torch.Tensor: RPN loss
        """

        objectness_loss = F.binary_cross_entropy_with_logits(objectness_pred, objectness_gt, reduction="mean")
        # filter out the background anchor bboxes
        filtered_bbox_pred = bbox_pred[objectness_gt == 1]  # (b, num_gt_objectness, 4)
        filtered_bbox_gt = bbox_gt[objectness_gt == 1]  # (b, num_gt_objectness, 4)
        bbox_loss = F.smooth_l1_loss(filtered_bbox_pred, filtered_bbox_gt, reduction="sum")

        return {
            "rpn_objectness_loss": lambda_rpn_objectness * objectness_loss,
            "rpn_bbox_loss": lambda_rpn_bbox * bbox_loss,
            "rpn_total_loss": lambda_rpn_objectness * objectness_loss + lambda_rpn_bbox * bbox_loss,
        }


class FastRCNNLoss(nn.Module):
    def __init__(self, background_class_index):
        super().__init__()
        self.background_class_index = background_class_index

    def forward(
        self,
        cls_pred: torch.Tensor,
        bbox_pred: torch.Tensor,
        cls_label: torch.Tensor,
        bbox_label: torch.Tensor,
        lambda_fast_rcnn_cls: float,
        lambda_fast_rcnn_bbox: float,
    ) -> dict[str, torch.Tensor]:
        """Compute the FastRCNN loss.

        Args:
            cls_pred (torch.Tensor): FastRCNN cls pred of shape (b, num_rois, num_classes)
            bbox_pred (torch.Tensor): FastRCNN bbox pred of shape (b, num_rois, num_classes, 4)
            cls_label (torch.Tensor): FastRCNN cls label of shape (b, num_rois)
            bbox_label (torch.Tensor): FastRCNN bbox label of shape (b, num_rois, num_classes, 4)
            lambda_ (int, optional): Param weighting cls_loss and bbox_loss. Defaults to 10.

        Returns:
            torch.Tensor: FastRCNN loss
        """

        cls_loss = F.cross_entropy(cls_pred, cls_label, reduction="mean")

        filtered_bbox_pred = bbox_pred[cls_label != self.background_class_index]  # (b, num_gt_cls, 4)
        filtered_bbox_gt = bbox_label[cls_label != self.background_class_index]  # (b, num_gt_cls, 4)
        bbox_loss = F.smooth_l1_loss(filtered_bbox_pred, filtered_bbox_gt, reduction="sum")

        return {
            "fast_rcnn_cls_loss": lambda_fast_rcnn_cls * cls_loss,
            "fast_rcnn_bbox_loss": lambda_fast_rcnn_bbox * bbox_loss,
            "fast_rcnn_total_loss": lambda_fast_rcnn_cls * cls_loss + lambda_fast_rcnn_bbox * bbox_loss,
        }


class FasterRCNNLoss(nn.Module):
    def __init__(self, background_class_idx):
        super().__init__()
        self.background_class_idx = background_class_idx
        self.rpn_loss = RPNLoss()
        self.fast_rcnn_loss = FastRCNNLoss(background_class_idx)

    def forward(
        self,
        rpn_objectness_pred: torch.Tensor,
        rpn_bbox_pred: torch.Tensor,
        # rpn_bbox_anchors: torch.Tensor,
        fast_rcnn_cls_pred: torch.Tensor,
        fast_rcnn_bbox_pred: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_bboxes: torch.Tensor,
        match_iou_threshold: float = 0.7,
        lambda_rpn_objectness=1,
        lambda_rpn_bbox=10,
        lambda_fast_rcnn_cls=10,
        lambda_fast_rcnn_bbox=10,
    ) -> torch.Tensor:
        """Compute the FasterRCNN loss which comprises of RPN loss and FastRCNN loss.

        For the RPN loss, we compute the IOU between the anchors and gt bboxes and assign each anchor box's prediction to the gt bbox with the highest IOU.
        For the FastRCNN loss, we compute the IOU between the pred bboxes and gt bboxes and assign each pred bbox to the gt bbox with the highest IOU.

        Args:
            objectness_pred: (B, nBB*1/pos_to_neg_ratio)
            rpn_bboxes_pred: (B, nBB*1/pos_to_neg_ratio, 4)
            rpn_bbox_anchors: (B, nBB*1/pos_to_neg_ratio, 4)
            foreground_objectness_pred: (B, nBB*pos_to_neg_ratio)
            foreground_bboxes_pred: (B, k*pos_to_neg_ratio, 4)
            fast_rcnn_cls_pred: (B, nBB, num_classes)
            fast_rcnn_bboxes_pred: (B, nBB, classes, 4)
            gt_cls (torch.Tensor): FastRCNN cls label of shape (b, gt_cls)
            gt_bboxes (torch.Tensor): FastRCNN bbox label of shape (b, gt_cls, 4)

        Returns:
            torch.Tensor: FasterRCNN loss
        """

        # you should match rpn_bbox_anchors with gt_bboxes early on for more training stability.
        # however, in an effort to simplify this pipeline, i'll match rpn_bbox_pred with gt_bboxes for now.
        # EFF: if the training is unstable, i'll match rpn_bbox_anchors with gt_bboxes.
        is_foreground, anchor_matches = self.match_based_on_iou(
            rpn_bbox_pred, gt_bboxes, match_iou_threshold
        )  # (b, nBB*1/pos_to_neg_ratio), (b, nBB*1/pos_to_neg_ratio)

        row_indices = torch.arange(gt_cls.shape[0]).unsqueeze(1).expand_as(gt_cls)
        # turn into 0s and 1s. how 0s and 1s is based on ious.
        rpn_objectness_gt = is_foreground.float()  # (b, nBB*1/pos_to_neg_ratio)
        # even when the match is background, we still find a match between the anchor and the gt bbox.
        rpn_bboxes_gt = gt_bboxes[row_indices, anchor_matches]  # (b, nBB*1/pos_to_neg_ratio, 4)

        row_indices = torch.arange(gt_cls.shape[0]).unsqueeze(1).expand_as(gt_cls)
        fast_rcnn_bbox_pred = fast_rcnn_bbox_pred[row_indices, anchor_matches]  # (b, nBB, 4)

        is_foreground, fast_rcnn_classifier_matches = self.match_based_on_iou(
            fast_rcnn_bbox_pred, gt_bboxes, match_iou_threshold
        )  # (b, nBB), (b, nBB)

        # background targets are set to the background class index. foreground targets are set to the gt class.
        fast_rcnn_cls_gt = (~is_foreground & torch.full_like(fast_rcnn_classifier_matches, self.background_class_idx)) + is_foreground & gt_cls[
            row_indices, fast_rcnn_classifier_matches
        ]  # (b, nBB)
        fast_rcnn_bboxes_gt = gt_bboxes[row_indices, fast_rcnn_classifier_matches]  # (b, nBB, 4)

        rpn_loss_dict = self.rpn_loss(rpn_objectness_pred, rpn_bbox_pred, rpn_objectness_gt, rpn_bboxes_gt, lambda_rpn_objectness, lambda_rpn_bbox)
        fast_rcnn_loss_dict = self.fast_rcnn_loss(
            fast_rcnn_cls_pred, fast_rcnn_bbox_pred, fast_rcnn_cls_gt, fast_rcnn_bboxes_gt, lambda_fast_rcnn_cls, lambda_fast_rcnn_bbox
        )

        faster_rcnn_loss = rpn_loss_dict["rpn_total_loss"] + fast_rcnn_loss_dict["fast_rcnn_total_loss"]

        return rpn_loss_dict | fast_rcnn_loss_dict | {"faster_rcnn_total_loss": faster_rcnn_loss}

    def match_based_on_iou(self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, iou_threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Match the pred bboxes to the gt bboxes based on the IOU.

        if iou < iou_threshold, then the pred bbox is background.

        Args:
            pred_bboxes (torch.Tensor): pred bboxes of shape (b, z1, 4)
            gt_bboxes (torch.Tensor): gt bboxes of shape (b, z2, 4)
            iou_threshold (float): threshold to consider a match

        Returns:
            torch.Tensor: matched indices of shape (b, nBB)
        """
        # compute iou between all pred bboxes and gt bboxes for each image in the batch
        ious = torch.vmap(torchvision.ops.box_iou)(pred_bboxes, gt_bboxes)  # (b, z1, z2)

        best_iou, best_gt_index = torch.argmax(ious, dim=2)  # (b, z1), (b, z1)
        is_foreground = best_iou > iou_threshold  # (b, z1)

        # think about how to set background classes to 0.

        # best_iou looks like
        # [[0.1, 0.2, 0.3, 0.4],
        #  [0.5, 0.6, 0.7, 0.8]]
        # best_gt_index looks like
        # [[0, 1, 2, 3],
        #  [0, 1, 2, 3]]
        # is_foreground looks like
        # [[False, False, False, True],
        #  [True, True, True, True]]

        return is_foreground, best_gt_index
