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
        fast_rcnn_bbox_offsets_pred: list[torch.Tensor],
        fast_rcnn_cls_gt_nms_fg_and_bg_some: list[torch.Tensor],
        fast_rcnn_bbox_offsets_gt: list[torch.Tensor],
        lambda_fast_rcnn_cls: float,
        lambda_fast_rcnn_bbox: float,
        *,
        device: str,
    ) -> dict[str, torch.Tensor]:
        total_bbox_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        num_images = len(fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox)
        num_cls_pred = 0
        num_bbox_pred = 0
        for image_idx in range(num_images):
            cls_loss = F.cross_entropy(
                fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox[image_idx], fast_rcnn_cls_gt_nms_fg_and_bg_some[image_idx], reduction="sum"
            )

            fg_bbox_indices = fast_rcnn_cls_gt_nms_fg_and_bg_some[image_idx] != self.background_class_index
            fg_bbox_offsets_pred = fast_rcnn_bbox_offsets_pred[image_idx][fg_bbox_indices]
            fg_bbox_offsets_gt = fast_rcnn_bbox_offsets_gt[image_idx][fg_bbox_indices]

            # if no fg bbox, don't compute the bbox loss.
            if fg_bbox_offsets_pred.numel() != 0:
                bbox_loss = F.smooth_l1_loss(fg_bbox_offsets_pred, fg_bbox_offsets_gt, reduction="sum")
                total_bbox_loss += bbox_loss
                num_bbox_pred += len(fg_bbox_offsets_pred)

            total_cls_loss += cls_loss
            num_cls_pred += len(fast_rcnn_cls_gt_nms_fg_and_bg_some[image_idx])

        total_cls_loss = total_cls_loss / (num_cls_pred + 1)
        total_bbox_loss = total_bbox_loss / (num_bbox_pred + 1)

        return {
            "fast_rcnn_cls_loss": lambda_fast_rcnn_cls * total_cls_loss,
            "fast_rcnn_bbox_loss": lambda_fast_rcnn_bbox * total_bbox_loss,
            "fast_rcnn_total_loss": lambda_fast_rcnn_cls * total_cls_loss + lambda_fast_rcnn_bbox * total_bbox_loss,
        }
