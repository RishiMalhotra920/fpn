import torch
import torch.nn.functional as F
from torch import nn

from fpn.YOLO_metrics import YOLOMetrics


class RPNLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        objectness_pred: torch.Tensor,
        bbox_pred: torch.Tensor,
        objectness_gt: torch.Tensor,
        bbox_gt: torch.Tensor,
        lambda_rpn_objectness=1,
        lambda_rpn_bbox=10,
    ) -> dict[str, torch.Tensor]:
        return super().__call__(
            objectness_pred,
            bbox_pred,
            objectness_gt,
            bbox_gt,
            lambda_rpn_objectness=lambda_rpn_objectness,
            lambda_rpn_bbox=lambda_rpn_bbox,
        )

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


        assert not torch.isnan(objectness_pred).any(), "NaN found in objectness_pred"
        assert not torch.isnan(bbox_pred).any(), "NaN found in bbox_pred"
        assert not torch.isnan(objectness_gt).any(), "NaN found in objectness_gt"
        assert not torch.isnan(bbox_gt).any(), "NaN found in bbox_gt"

        assert not torch.isinf(objectness_pred).any(), "Inf found in objectness_pred"
        assert not torch.isinf(bbox_pred).any(), "Inf found in bbox_pred"
        assert not torch.isinf(objectness_gt).any(), "Inf found in objectness_gt"
        assert not torch.isinf(bbox_gt).any(), "Inf found in bbox_gt"
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
        # apply_offsets_to_fast_rcnn_bboxes(

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
        list_of_picked_bboxes_gt_matches: list[torch.Tensor],
        is_fast_rcnn_preds_foreground: list[torch.Tensor],
        gt_cls: torch.Tensor,
        gt_bboxes: torch.Tensor,
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
            list_of_picked_bboxes_gt_matches,
            is_fast_rcnn_preds_foreground,
            gt_cls,
            gt_bboxes,
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
        list_of_picked_bboxes_gt_matches: list[torch.Tensor],
        is_fast_rcnn_preds_foreground: list[torch.Tensor],
        gt_cls: torch.Tensor,
        gt_bboxes: torch.Tensor,
        *,
        lambda_rpn_objectness=1,
        lambda_rpn_bbox=10,
        lambda_fast_rcnn_cls=10,
        lambda_fast_rcnn_bbox=10,
    ) -> dict[str, torch.Tensor]:
        """Compute the FasterRCNN loss which comprises of RPN loss and FastRCNN loss.

        For the RPN loss, we compute the IOU between the anchors and gt bboxes and assign each anchor box's prediction to the gt bbox with the highest IOU.
        For the FastRCNN loss, we compute the IOU between the pred bboxes and gt bboxes and assign each pred bbox to the gt bbox with the highest IOU.


        # rpn_objectness_pred: list[num_pyramids, torch.Tensor(b, s*s*9)]
        # fast_rcnn_bbox_pred: list[num_pyramids, torch.Tensor(b, 3*s*s*9, 4)]
        # fast_rcnn_cls_pred: list[num_pyramids, list[num_images, torch.Tensor(L_i)]
        # fast_rcnn_bbox_pred: list[num_pyramids, list[num_images, torch.Tensor(L_i, 4)]
        # where L_i is the number of picked bounding boxes in image i

        # gt_cls: torch.Tensor(b, max_gt_bboxes) where there are nBB boxes padded with 0s.
        # gt_bboxes: torch.Tensor(b, max_gt_bboxes, 4) where there are nBB boxes padded with 0s.
        # num_gt_bboxes_in_each_image: torch.Tensor(b) where each element is the number of gt bboxes in each image.


        Args:
        Returns:
            torch.Tensor: FasterRCNN loss
        """

        # for faster_rcnn_cls_pred, faster_rcnn_bbox_pred in zip(fast_rcnn_cls_pred, fast_rcnn_bbox_pred):
        # for pred in range(3):

        # you should match rpn_bbox_anchors with gt_bboxes early on for more training stability.
        # however, in an effort to simplify this pipeline, i'll match rpn_bbox_pred with gt_bboxes for now.
        # EFF: if the training is unstable, i'll match rpn_bbox_anchors with gt_bboxes.
        # is_rpn_preds_foreground, anchor_matches = self.match_based_on_iou(
        # rpn_bbox_pred, gt_bboxes, match_iou_threshold
        # )  # (b, nBB*1/pos_to_neg_ratio), (b, nBB*1/pos_to_neg_ratio)

        # row_indices = torch.arange(gt_cls.shape[0]).unsqueeze(1).expand_as(gt_cls)
        # turn into 0s and 1s. how 0s and 1s is based on ious.
        # rpn_objectness_gt = is_foreground.float()  # (b, nBB*1/pos_to_neg_ratio)

        # even when the match is background, we still find a match between the anchor and the gt bbox.
        # rpn_bboxes_gt = gt_bboxes[row_indices, anchor_matches]  # (b, nBB*1/pos_to_neg_ratio, 4)
        rpn_bboxes_gt = [gt_bboxes[i, rpn_bbox_matches[i]] for i in range(gt_bboxes.shape[0])]

        # row_indices = torch.arange(gt_cls.shape[0]).unsqueeze(1).expand_as(gt_cls)
        # sit down and implement loss function with fast_rcnn_cls_pred and fast_rcnn_bbox_pred

        # fast_rcnn_cls_pred = torch.cat(fast_rcnn_cls_pred, dim=1)  # (b, nBB, num_classes)
        # fast_rcnn_bbox_pred = fast_rcnn_bbox_pred[row_indices, anchor_matches]  # (b, nBB, 4)

        # is_fast_rcnn_preds_foreground, fast_rcnn_matches = self.match_based_on_iou(
        #     # fast_rcnn_cls_pred,
        #     fast_rcnn_bbox_pred,
        #     gt_bboxes,
        #     match_iou_threshold,
        # )  # (b, nBB), (b, nBB)

        # print('devices', is_fast_rcnn_preds_foreground[0].device, list_of_picked_bboxes_gt_matches[0].device, gt_cls.device)
        # exit()
        fast_rcnn_cls_gt = [
            (
                (~is_image_preds_foreground & torch.full_like(is_image_preds_foreground, self.background_class_idx, device=self.device))
                + (is_image_preds_foreground & gt_cls[i, is_image_preds_foreground])
            )
            for i, (is_image_preds_foreground, image_picked_bboxes_gt_matches) in enumerate(
                zip(is_fast_rcnn_preds_foreground, list_of_picked_bboxes_gt_matches)
            )
        ]

        fast_rcnn_bboxes_gt = []
        fast_rcnn_bboxes_max_class_pred = []
        for i in range(len(list_of_picked_bboxes_gt_matches)):
            picked_bboxes_gt_matches = list_of_picked_bboxes_gt_matches[i]
            fast_rcnn_bboxes_gt.append(gt_bboxes[i, picked_bboxes_gt_matches])
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
        
            fast_rcnn_bboxes_max_class_pred.append(fast_rcnn_bbox_pred[i][row_indices, max_class_pred_index])

        # fast_rcnn_bboxes_gt = [gt_bboxes[i, picked_bboxes_gt_matches] for i, picked_bboxes_gt_matches in enumerate(list_of_picked_bboxes_gt_matches)]

        # fast_rcnn_bboxes_gt = gt_bboxes[row_indices, fast_rcnn_matches]  # (b, nBB, 4)

        # for rpn_loss, we can actually stack up the list of tensors into one tensor since each list element is of the same shape

        rpn_loss_dict = self.rpn_loss(
            rpn_objectness_pred,
            rpn_bbox_pred,
            torch.stack(is_rpn_preds_foreground).float(),
            torch.stack(rpn_bboxes_gt),
            lambda_rpn_objectness,
            lambda_rpn_bbox,
        )

        fast_rcnn_loss_dict = self.fast_rcnn_loss(
            fast_rcnn_cls_pred, fast_rcnn_bboxes_max_class_pred, fast_rcnn_cls_gt, fast_rcnn_bboxes_gt, lambda_fast_rcnn_cls, lambda_fast_rcnn_bbox
        )

        faster_rcnn_loss = rpn_loss_dict["rpn_total_loss"] + fast_rcnn_loss_dict["fast_rcnn_total_loss"]

        # self.metric.compute_values(fast_rcnn_cls_pred, fast_rcnn_bboxes_max_class_pred, fast_rcnn_cls_gt, fast_rcnn_bboxes_gt

        return rpn_loss_dict | fast_rcnn_loss_dict | {"faster_rcnn_total_loss": faster_rcnn_loss}
