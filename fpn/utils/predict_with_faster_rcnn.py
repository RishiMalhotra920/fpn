import torch

from fpn.models import FPN, FasterRCNN
from fpn.utils.anchor_utils import create_anchors


def predict_with_faster_rcnn(
    images: torch.Tensor,
    image_dim: int,
    backbone: FPN,
    faster_rcnn_model: FasterRCNN,
    raw_cls_gt: torch.Tensor,
    raw_bbox_gt: torch.Tensor,
    *,
    device: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    all_anchor_widths, all_anchor_heights, all_anchor_positions = create_anchors((image_dim, image_dim), device)

    with torch.no_grad():
        fpn_maps = backbone(images)

        all_rpn_bbox_pred = []
        all_rpn_label_pred = []

        for fpn_map, anchor_heights, anchor_widths, anchor_positions in zip(fpn_maps, all_anchor_heights, all_anchor_widths, all_anchor_positions):
            (
                rpn_objectness_pred,
                rpn_bbox_offset_pred,
                rpn_bbox_pred_nms_fg_and_bg_some,
                rpn_bbox_pred_nms_fg_and_bg_some_fg_mask,
                # fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox,
                # fast_rcnn_bbox_offsets_pred,
                rpn_objectness_gt,
                rpn_bbox_offset_gt,
                # fast_rcnn_cls_gt_nms_fg_and_bg_some,
                # fast_rcnn_bbox_offsets_gt,
                # rpn_num_fg_bbox_picked,
                # rpn_num_bg_bbox_picked,
            ) = faster_rcnn_model(fpn_map, anchor_heights, anchor_widths, anchor_positions, raw_cls_gt, raw_bbox_gt)

            all_rpn_bbox_pred.append(rpn_bbox_pred_nms_fg_and_bg_some)  # list[b, torch.Tensor(L_i, 4)]
            all_rpn_label_pred.append(rpn_bbox_pred_nms_fg_and_bg_some_fg_mask)  # list[b, torch.Tensor(L_i, 4)]

        # all_rpn_bbox_pred: list[3, list[b, torch.Tensor(L_i, 4)]]
        # all_rpn_cls_pred: list[3, list[b, torch.Tensor(L_i, 4)]]

        rpn_pred_bbox_by_image = [
            torch.cat((t1, t2, t3), dim=0) for t1, t2, t3 in zip(all_rpn_bbox_pred[0], all_rpn_bbox_pred[1], all_rpn_bbox_pred[2])
        ]
        rpn_pred_label_by_image = [
            torch.cat((t1, t2, t3), dim=0) for t1, t2, t3 in zip(all_rpn_label_pred[0], all_rpn_label_pred[1], all_rpn_label_pred[2])
        ]

    return rpn_pred_label_by_image, rpn_pred_bbox_by_image
