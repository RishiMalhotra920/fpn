import torch
from torch import nn

from fpn.loss.fast_rcnn_loss import FastRCNNLoss
from fpn.loss.rpn_loss import RPNLoss
from fpn.YOLO_metrics import YOLOMetrics


class FasterRCNNLoss(nn.Module):
    def __init__(self, background_class_idx: int, device: str):
        super().__init__()
        self.background_class_idx = background_class_idx
        self.rpn_loss = RPNLoss()
        self.fast_rcnn_loss = FastRCNNLoss(background_class_idx)
        self.metric = YOLOMetrics()
        self.device = device

    def __call__(
        self,
        rpn_objectness_pred: torch.Tensor,
        rpn_bbox_pred: torch.Tensor,
        fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox: list[torch.Tensor],
        fast_rcnn_bbox_pred_for_some_rpn_bbox: list[torch.Tensor],
        rpn_objectness_gt: torch.Tensor,
        rpn_bbox_gt: torch.Tensor,
        fast_rcnn_cls_gt_nms_fg_and_bg_some: list[torch.Tensor],
        fast_rcnn_bbox_gt_nms_fg_and_bg_some: list[torch.Tensor],
        *,
        device: str,
        lambda_rpn_objectness=1,
        lambda_rpn_bbox=10,
        lambda_fast_rcnn_cls=10,
        lambda_fast_rcnn_bbox=10,
    ) -> dict[str, torch.Tensor]:
        return super().__call__(
            rpn_objectness_pred,
            rpn_bbox_pred,
            fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox,
            fast_rcnn_bbox_pred_for_some_rpn_bbox,
            rpn_objectness_gt,
            rpn_bbox_gt,
            fast_rcnn_cls_gt_nms_fg_and_bg_some,
            fast_rcnn_bbox_gt_nms_fg_and_bg_some,
            device=device,
            lambda_rpn_objectness=lambda_rpn_objectness,
            lambda_rpn_bbox=lambda_rpn_bbox,
            lambda_fast_rcnn_cls=lambda_fast_rcnn_cls,
            lambda_fast_rcnn_bbox=lambda_fast_rcnn_bbox,
        )

    def forward(
        self,
        rpn_objectness_pred: torch.Tensor,
        rpn_bbox_pred: torch.Tensor,
        fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox: list[torch.Tensor],
        fast_rcnn_bbox_pred_for_some_rpn_bbox: list[torch.Tensor],
        rpn_objectness_gt: torch.Tensor,
        rpn_bbox_gt: torch.Tensor,
        fast_rcnn_cls_gt_nms_fg_and_bg_some: list[torch.Tensor],
        fast_rcnn_bbox_gt_nms_fg_and_bg_some: list[torch.Tensor],
        *,
        device: str,
        lambda_rpn_objectness=1,
        lambda_rpn_bbox=10,
        lambda_fast_rcnn_cls=10,
        lambda_fast_rcnn_bbox=10,
    ) -> dict[str, torch.Tensor]:
        """Compute the FasterRCNN loss which comprises of RPN loss and FastRCNN loss.

        For the RPN loss, we compute the IOU between the anchors and gt bbox and assign each anchor box's prediction to the gt bbox with the highest IOU.
        For the FastRCNN loss, we compute the IOU between the pred bbox and gt bbox and assign each pred bbox to the gt bbox with the highest IOU.


        # rpn_objectness_pred: list[num_pyramids, torch.Tensor(b, s*s*9)]
        # fast_rcnn_bbox_pred: list[num_pyramids, torch.Tensor(b, 3*s*s*9, 4)]
        # fast_rcnn_cls_pred: list[num_pyramids, list[num_images, torch.Tensor(L_i)]
        # fast_rcnn_bbox_pred: list[num_pyramids, list[num_images, torch.Tensor(L_i, 4)]
        # where L_i is the number of picked bounding boxes in image i

        # gt_cls: torch.Tensor(b, max_gt_bbox) where there are nBB boxes padded with 0s.
        # gt_bbox: torch.Tensor(b, max_gt_bbox, 4) where there are nBB boxes padded with 0s.
        # num_gt_bbox_in_each_image: torch.Tensor(b) where each element is the number of gt bbox in each image.


        Args:
        Returns:
            torch.Tensor: FasterRCNN loss
        """

        rpn_loss_dict = self.rpn_loss(
            rpn_objectness_pred,
            rpn_bbox_pred,
            rpn_objectness_gt,
            rpn_bbox_gt,
            lambda_rpn_objectness=lambda_rpn_objectness,
            lambda_rpn_bbox=lambda_rpn_bbox,
            device=device,
        )

        fast_rcnn_loss_dict = self.fast_rcnn_loss(
            fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox,
            fast_rcnn_bbox_pred_for_some_rpn_bbox,
            fast_rcnn_cls_gt_nms_fg_and_bg_some,
            fast_rcnn_bbox_gt_nms_fg_and_bg_some,
            lambda_fast_rcnn_cls=lambda_fast_rcnn_cls,
            lambda_fast_rcnn_bbox=lambda_fast_rcnn_bbox,
            device=device,
        )

        faster_rcnn_loss = rpn_loss_dict["rpn_total_loss"] + fast_rcnn_loss_dict["fast_rcnn_total_loss"]

        # self.metric.compute_values(fast_rcnn_cls_pred, fast_rcnn_bbox_max_class_pred, fast_rcnn_cls_gt, fast_rcnn_bbox_gt

        return rpn_loss_dict | fast_rcnn_loss_dict | {"faster_rcnn_total_loss": faster_rcnn_loss}
