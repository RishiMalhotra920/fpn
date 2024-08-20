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
    ) -> dict[str, torch.Tensor]:
        return super().__call__(
            cls_pred,
            bbox_pred,
            cls_label,
            bbox_label,
            lambda_fast_rcnn_cls=lambda_fast_rcnn_cls,
            lambda_fast_rcnn_bbox=lambda_fast_rcnn_bbox,
        )

    def forward(
        self,
        cls_pred: list[torch.Tensor],
        bbox_pred: list[torch.Tensor],
        cls_label: list[torch.Tensor],
        bbox_label: list[torch.Tensor],
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
        # apply_offsets_to_fast_rcnn_bbox(

        # cls_loss = torch.mean(torch.tensor([for i in range(len(cls_pred))]))

        # cls_loss = F.cross_entropy(cls_pred, cls_label, reduction="mean")

        # filtered_bbox_pred = bbox_pred[cls_label != self.background_class_index]  # (b, num_gt_cls, 4)
        # filtered_bbox_gt = bbox_label[cls_label != self.background_class_index]  # (b, num_gt_cls, 4)

        total_bbox_loss = torch.tensor(0.0, device=cls_pred[0].device)
        total_cls_loss = torch.tensor(0.0, device=cls_pred[0].device)
        for image_idx in range(len(cls_label)):
            cls_loss = F.cross_entropy(cls_pred[image_idx], cls_label[image_idx], reduction="mean")
            total_cls_loss += cls_loss
            filtered_image_bbox_pred = bbox_pred[image_idx][cls_label[image_idx] != self.background_class_index]
            filtered_image_bbox_gt = bbox_label[image_idx][cls_label[image_idx] != self.background_class_index]

            bbox_loss = F.smooth_l1_loss(filtered_image_bbox_pred, filtered_image_bbox_gt, reduction="sum")
            total_bbox_loss += bbox_loss

        return {
            "fast_rcnn_cls_loss": lambda_fast_rcnn_cls * total_cls_loss,
            "fast_rcnn_bbox_loss": lambda_fast_rcnn_bbox * total_bbox_loss,
            "fast_rcnn_total_loss": lambda_fast_rcnn_cls * total_cls_loss + lambda_fast_rcnn_bbox * total_bbox_loss,
        }
