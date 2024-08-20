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
        is_rpn_preds_foreground: list[torch.Tensor],
        rpn_bbox_matches: list[torch.Tensor],
        # rpn_bbox_anchors: torch.Tensor,
        fast_rcnn_cls_pred: list[torch.Tensor],
        fast_rcnn_bbox_pred: list[torch.Tensor],
        list_of_picked_bbox_gt_matches: list[torch.Tensor],
        is_fast_rcnn_preds_foreground: list[torch.Tensor],
        gt_cls: torch.Tensor,
        gt_bbox: torch.Tensor,
        *,
        lambda_rpn_objectness=1,
        lambda_rpn_bbox=10,
        lambda_fast_rcnn_cls=10,
        lambda_fast_rcnn_bbox=10,
    ) -> dict[str, torch.Tensor]:
        return super().__call__(
            rpn_objectness_pred,
            rpn_bbox_pred,
            is_rpn_preds_foreground,
            rpn_bbox_matches,
            fast_rcnn_cls_pred,
            fast_rcnn_bbox_pred,
            list_of_picked_bbox_gt_matches,
            is_fast_rcnn_preds_foreground,
            gt_cls,
            gt_bbox,
            lambda_rpn_objectness=lambda_rpn_objectness,
            lambda_rpn_bbox=lambda_rpn_bbox,
            lambda_fast_rcnn_cls=lambda_fast_rcnn_cls,
            lambda_fast_rcnn_bbox=lambda_fast_rcnn_bbox,
        )

    def forward(
        self,
        rpn_objectness_pred: torch.Tensor,
        rpn_bbox_pred: torch.Tensor,
        is_rpn_preds_foreground: list[torch.Tensor],
        rpn_bbox_matches: list[torch.Tensor],
        # rpn_bbox_anchors: torch.Tensor,
        fast_rcnn_cls_pred: list[torch.Tensor],
        fast_rcnn_bbox_pred: list[torch.Tensor],
        list_of_picked_bbox_gt_matches: list[torch.Tensor],
        is_fast_rcnn_preds_foreground: list[torch.Tensor],
        gt_cls: torch.Tensor,
        gt_bbox: torch.Tensor,
        *,
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

        # all of these should be calculated in forward() not in loss()
        rpn_bbox_gt = [gt_bbox[i, rpn_bbox_matches[i]] for i in range(gt_bbox.shape[0])]

        fast_rcnn_cls_gt = [
            (
                (~is_image_preds_foreground & torch.full_like(is_image_preds_foreground, self.background_class_idx, device=self.device))
                + (is_image_preds_foreground & gt_cls[i, is_image_preds_foreground])
            )
            for i, (is_image_preds_foreground, image_picked_bbox_gt_matches) in enumerate(
                zip(is_fast_rcnn_preds_foreground, list_of_picked_bbox_gt_matches)
            )
        ]

        fast_rcnn_bbox_gt = []
        fast_rcnn_bbox_max_class_pred = []
        for i in range(len(list_of_picked_bbox_gt_matches)):
            picked_bbox_gt_matches = list_of_picked_bbox_gt_matches[i]
            fast_rcnn_bbox_gt.append(gt_bbox[i, picked_bbox_gt_matches])
            row_indices = torch.arange(len(fast_rcnn_cls_gt[i]), device=self.device)
            max_class_pred_index = fast_rcnn_cls_gt[i]
            print(f"Image {i}:")
            print(list(max_class_pred_index))
            print(f"  fast_rcnn_bbox_pred[i] shape: {fast_rcnn_bbox_pred[i].shape}")
            print(f"  row_indices shape: {row_indices.shape}")
            print(f"  max_class_pred_index shape: {max_class_pred_index.shape}")
            print(f"  row_indices max: {row_indices.max().item()}")
            print(f"  max_class_pred_index max: {max_class_pred_index.max().item()}")
            print(f"  fast_rcnn_cls_gt[i] unique values: {fast_rcnn_cls_gt[i].unique()}")

            # Check if indices are within bounds
            assert row_indices.max() < fast_rcnn_bbox_pred[i].shape[0], "row_indices out of bounds"
            assert max_class_pred_index.max() < fast_rcnn_bbox_pred[i].shape[1], "max_class_pred_index out of bounds"

            fast_rcnn_bbox_max_class_pred.append(fast_rcnn_bbox_pred[i][row_indices, max_class_pred_index])

        rpn_loss_dict = self.rpn_loss(
            rpn_objectness_pred,
            rpn_bbox_pred,
            torch.stack(is_rpn_preds_foreground).float(),
            torch.stack(rpn_bbox_gt),
            lambda_rpn_objectness,
            lambda_rpn_bbox,
        )

        fast_rcnn_loss_dict = self.fast_rcnn_loss(
            fast_rcnn_cls_pred, fast_rcnn_bbox_max_class_pred, fast_rcnn_cls_gt, fast_rcnn_bbox_gt, lambda_fast_rcnn_cls, lambda_fast_rcnn_bbox
        )

        faster_rcnn_loss = rpn_loss_dict["rpn_total_loss"] + fast_rcnn_loss_dict["fast_rcnn_total_loss"]

        # self.metric.compute_values(fast_rcnn_cls_pred, fast_rcnn_bbox_max_class_pred, fast_rcnn_cls_gt, fast_rcnn_bbox_gt

        return rpn_loss_dict | fast_rcnn_loss_dict | {"faster_rcnn_total_loss": faster_rcnn_loss}
