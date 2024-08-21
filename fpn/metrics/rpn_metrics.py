import torch
import torchvision


def calculate_rpn_metrics(
    rpn_bbox_pred_nms_fg_and_bg_some: list[torch.Tensor], raw_bbox_gt: torch.Tensor, thresholds: list[float]
) -> dict[str, float]:
    # calculate iou @ threshold
    # note you can't change the name of rpn_recall@

    accum = {f"NumOverIoU@{t}": 0.0 for t in thresholds} | {f"Total@{t}": 0.0 for t in thresholds}

    for image_bbox_pred, image_bbox_gt in zip(rpn_bbox_pred_nms_fg_and_bg_some, raw_bbox_gt):
        if image_bbox_pred.numel() == 0 or image_bbox_gt.numel() == 0:
            continue

        ious = torchvision.ops.box_iou(image_bbox_gt, image_bbox_pred)  # (num_gt_boxes, num_pred_boxes)

        ious = ious.max(dim=1).values  # (num_gt_boxes,)

        for threshold in thresholds:
            num_bbox_over_threshold = (ious > threshold).sum().item()
            accum[f"NumOverIoU@{threshold}"] += num_bbox_over_threshold
            accum[f"Total@{threshold}"] += len(image_bbox_gt)

    result = {}
    for threshold in thresholds:
        if accum[f"Total@{threshold}"] == 0:
            result[f"rpn_recall@{threshold}"] = 1.0
        else:
            result[f"rpn_recall@{threshold}"] = accum[f"NumOverIoU@{threshold}"] / accum[f"Total@{threshold}"]

    return result
