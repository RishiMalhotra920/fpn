import torch
import torch.nn.functional as F
from torch import nn


class RPNLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        objectness_pred: torch.Tensor,
        bbox_pred: torch.Tensor,
        objectness_gt: torch.Tensor,
        bbox_gt: torch.Tensor,
        *,
        device: str,
        lambda_rpn_objectness=1,
        lambda_rpn_bbox=10,
    ) -> dict[str, torch.Tensor]:
        return super().__call__(
            objectness_pred,
            bbox_pred,
            objectness_gt,
            bbox_gt,
            device=device,
            lambda_rpn_objectness=lambda_rpn_objectness,
            lambda_rpn_bbox=lambda_rpn_bbox,
        )

    def forward(
        self,
        objectness_pred: torch.Tensor,
        bbox_offset_pred: torch.Tensor,
        objectness_gt: torch.Tensor,
        bbox_offset_gt: torch.Tensor,
        *,
        device: str,
        lambda_rpn_objectness=1,
        lambda_rpn_bbox=10,
    ) -> dict[str, torch.Tensor]:
        """Compute the RPN loss.

        objectness_pred are all the objectness predictions and objectness_gt are all the ground truth objectness matched to the anchors.

        Before computing the bbox loss, we filter out the background anchor bbox. We only compute the bbox loss for the foreground anchor bbox.
        bbox_pred are all the bbox predictions. bbox_gt are all the ground truth bbox matched to the anchors.

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

        filtered_bbox_pred = bbox_offset_pred[objectness_gt == 1]  # (b, num_gt_objectness, 4)
        filtered_bbox_gt = bbox_offset_gt[objectness_gt == 1]  # (b, num_gt_objectness, 4)
        bbox_loss = F.smooth_l1_loss(filtered_bbox_pred, filtered_bbox_gt, reduction="mean")

        return {
            "rpn_objectness_loss": lambda_rpn_objectness * objectness_loss,
            "rpn_bbox_loss": lambda_rpn_bbox * bbox_loss,
            "rpn_total_loss": lambda_rpn_objectness * objectness_loss + lambda_rpn_bbox * bbox_loss,
        }
