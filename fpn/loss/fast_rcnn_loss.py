import torch
from torch import nn
from torch.nn import functional as F


class FastRCNNLoss(nn.Module):
    def __init__(self, background_class_index):
        super().__init__()
        self.background_class_index = background_class_index

    def __call__(
        self,
        cls_pred: list[torch.Tensor],
        bbox_pred: list[torch.Tensor],
        cls_label: list[torch.Tensor],
        bbox_label: list[torch.Tensor],
        lambda_fast_rcnn_cls: float,
        lambda_fast_rcnn_bbox: float,
        *,
        device: str,
    ) -> dict[str, torch.Tensor]:
        return super().__call__(
            cls_pred,
            bbox_pred,
            cls_label,
            bbox_label,
            lambda_fast_rcnn_cls=lambda_fast_rcnn_cls,
            lambda_fast_rcnn_bbox=lambda_fast_rcnn_bbox,
            device=device,
        )

    def forward(
        self,
        fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox: list[torch.Tensor],
        fast_rcnn_bbox_pred_for_some_rpn_bbox: list[torch.Tensor],
        fast_rcnn_cls_gt_nms_fg_and_bg_some: list[torch.Tensor],
        fast_rcnn_bbox_gt_nms_fg_and_bg_some: list[torch.Tensor],
        lambda_fast_rcnn_cls: float,
        lambda_fast_rcnn_bbox: float,
        *,
        device: str,
    ) -> dict[str, torch.Tensor]:
        total_bbox_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        num_images = len(fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox)
        for image_idx in range(num_images):
            cls_loss = F.cross_entropy(
                fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox[image_idx], fast_rcnn_cls_gt_nms_fg_and_bg_some[image_idx], reduction="mean"
            )

            fg_bbox_indices = fast_rcnn_cls_gt_nms_fg_and_bg_some[image_idx] != self.background_class_index
            filtered_image_bbox_pred = fast_rcnn_bbox_pred_for_some_rpn_bbox[image_idx][fg_bbox_indices]
            filtered_image_bbox_gt = fast_rcnn_bbox_gt_nms_fg_and_bg_some[image_idx][fg_bbox_indices]

            bbox_loss = F.smooth_l1_loss(filtered_image_bbox_pred, filtered_image_bbox_gt, reduction="mean")

            total_cls_loss += cls_loss
            total_bbox_loss += bbox_loss

        return {
            "fast_rcnn_cls_loss": lambda_fast_rcnn_cls * total_cls_loss,
            "fast_rcnn_bbox_loss": lambda_fast_rcnn_bbox * total_bbox_loss,
            "fast_rcnn_total_loss": lambda_fast_rcnn_cls * total_cls_loss + lambda_fast_rcnn_bbox * total_bbox_loss,
        }
