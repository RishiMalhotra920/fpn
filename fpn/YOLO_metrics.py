import numpy as np
import torch

from fpn.utils.compute_iou import compute_iou


class YOLOMetrics:
    def __init__(self):
        self.CLASS_INDEX = 4
        self.CONF_INDEX = 5

    def compute_values(
        self,
        fast_rcnn_cls_pred: list[torch.Tensor],
        fast_rcnn_bboxes_pred: list[torch.Tensor],
        fast_rcnn_cls_gt: torch.Tensor,
        fast_rcnn_bboxes_gt: torch.Tensor,
    ) -> dict:
        """Computes YOLO percent correctness according to the YOLOv1 paper."""

        total_correct, total_localization, total_other, total_background, total_pred_bboxes = 0, 0, 0, 0, 0

        # EFF: not efficient but works
        pred = np.concatenate((fast_rcnn_cls_pred, fast_rcnn_bboxes_pred), axis=1)
        gt = np.concatenate((fast_rcnn_cls_gt, fast_rcnn_bboxes_gt), axis=1)

        # EFF: not efficient to for loop over images but works...
        for image_idx in range(len(pred)):
            pred_image = pred[image_idx]
            gt_image = gt[image_idx]

            # no bounding box predictions in the image
            if len(pred_image) == 0:
                continue
            else:
                print("trace 0")
                correct, localization, other, background, total_pred_bboxes_in_image = self._get_yolo_metrics_from_boxes_in_image(
                    pred_image, gt_image
                )
                print("correct are", correct)
                total_correct += correct
                total_localization += localization
                total_other += other
                total_background += background
                total_pred_bboxes += total_pred_bboxes_in_image

        return {
            "num_correct": total_correct,
            "num_incorrect_localization": total_localization,
            "num_incorrect_other": total_other,
            "num_incorrect_background": total_background,
            "num_objects": total_pred_bboxes,
        }

    def _get_yolo_metrics_from_boxes_in_image(self, pred_image: np.ndarray, gt_image: np.ndarray) -> tuple[int, int, int, int, int]:
        """"""

        order = np.argsort(pred_image[:, self.CONF_INDEX])[::-1]
        pred_image = pred_image[order]

        # num
        gt_boxes_match_order = np.full((len(gt_image), 2), -1.0)  # num_gt_boxes x [iou with pred box, class match]

        gt_image_indices = np.arange(0, len(gt_image))
        for i, pred_bbox in enumerate(pred_image):
            # compute ious with gt boxes that haven't been matched yet.
            unmatched_gt_boxes_mask = gt_boxes_match_order[:, 0] < 0
            if not np.any(unmatched_gt_boxes_mask):
                break
            print("this is unmatched_gt_boxes_mask", unmatched_gt_boxes_mask)
            ious = compute_iou(pred_bbox[np.newaxis, :4], gt_image[unmatched_gt_boxes_mask, :4])

            print("this is ious", ious)
            highest_iou_idx = np.argmax(ious)
            highest_iou = ious[highest_iou_idx]
            # apply the mask to the gt image indices and get the index of highest iou gt box from the indices array.
            highest_unmatched_gt_iou_idx = gt_image_indices[unmatched_gt_boxes_mask][highest_iou_idx]

            class_match = pred_bbox[self.CLASS_INDEX] == gt_image[highest_unmatched_gt_iou_idx, self.CLASS_INDEX]
            gt_boxes_match_order[highest_unmatched_gt_iou_idx, :] = [highest_iou, class_match]

        print("trace 1", gt_boxes_match_order)
        matched_gt_boxes = gt_boxes_match_order[gt_boxes_match_order[:, 0] >= 0]
        print("this is matched_gt_boxes", gt_boxes_match_order, matched_gt_boxes)

        matched_ious, matched_classes = matched_gt_boxes[:, 0], matched_gt_boxes[:, 1].astype(bool)
        print("this is matched_ious", matched_ious, matched_classes)
        correct = np.sum((matched_ious > 0.5) & matched_classes, dtype=int)
        localization = np.sum((0.1 < matched_ious) & (matched_ious < 0.5) & matched_classes, dtype=int)
        other = np.sum((matched_ious > 0.1) & ~matched_classes, dtype=int)
        # add the extra boxes in the prediction that don't match any gt box to the background count
        background = np.sum(matched_ious < 0.1, dtype=int) + max(0, len(pred_image) - len(gt_image))
        pred_boxes_in_image = len(pred_image)
        # a = np.sum(matched_ious > 0.5, dtype=int)
        print("this is localization", localization)

        return correct, localization, other, background, pred_boxes_in_image
