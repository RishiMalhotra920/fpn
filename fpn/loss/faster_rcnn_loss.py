import torch
from torch import nn

from fpn.loss.fast_rcnn_loss import FastRCNNLoss
from fpn.loss.rpn_loss import RPNLoss

# from fpn.YOLO_metrics import YOLOMetrics


class FasterRCNNLoss(nn.Module):
    def __init__(self, background_class_idx: int, device: str):
        super().__init__()
        self.background_class_idx = background_class_idx
        self.rpn_loss = RPNLoss()
        self.fast_rcnn_loss = FastRCNNLoss(background_class_idx)
        # self.metric = YOLOMetrics()
        self.device = device

    def __call__(
        self,
        rpn_objectness_pred: torch.Tensor,
        rpn_bbox_offset_pred: torch.Tensor,
        fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox: list[torch.Tensor],
        fast_rcnn_bbox_offsets_pred: list[torch.Tensor],
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
            rpn_bbox_offset_pred,
            fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox,
            fast_rcnn_bbox_offsets_pred,
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
        rpn_bbox_offset_pred: torch.Tensor,
        fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox: list[torch.Tensor],
        fast_rcnn_bbox_offsets_pred: list[torch.Tensor],
        rpn_objectness_gt: torch.Tensor,
        rpn_bbox_offset_gt: torch.Tensor,
        fast_rcnn_cls_gt_nms_fg_and_bg_some: list[torch.Tensor],
        fast_rcnn_bbox_offsets_gt: list[torch.Tensor],
        *,
        device: str,
        lambda_rpn_objectness=1,
        lambda_rpn_bbox=10,
        lambda_fast_rcnn_cls=10,
        lambda_fast_rcnn_bbox=10,
    ) -> dict[str, torch.Tensor]:
        rpn_loss_dict = self.rpn_loss(
            rpn_objectness_pred,
            rpn_bbox_offset_pred,
            rpn_objectness_gt,
            rpn_bbox_offset_gt,
            lambda_rpn_objectness=lambda_rpn_objectness,
            lambda_rpn_bbox=lambda_rpn_bbox,
            device=device,
        )

        fast_rcnn_loss_dict = self.fast_rcnn_loss(
            fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox,
            fast_rcnn_bbox_offsets_pred,
            fast_rcnn_cls_gt_nms_fg_and_bg_some,
            fast_rcnn_bbox_offsets_gt,
            lambda_fast_rcnn_cls=lambda_fast_rcnn_cls,
            lambda_fast_rcnn_bbox=lambda_fast_rcnn_bbox,
            device=device,
        )

        faster_rcnn_loss = rpn_loss_dict["rpn_total_loss"] + fast_rcnn_loss_dict["fast_rcnn_total_loss"]

        return rpn_loss_dict | fast_rcnn_loss_dict | {"faster_rcnn_total_loss": faster_rcnn_loss}
