import numpy as np

from fpn.evaluation.metrics.metric import Metric


class YOLOMetric(Metric):
    def __init__(self):
        self.CLASS_INDEX = 4
        self.CONF_INDEX = 5

    def compute_values(self, pred: list[np.ndarray], gt: list[np.ndarray]) -> tuple[float, float, float, float, float]:
        """Computes YOLO percent correctness according to the YOLOv1 paper.

        Args:
            pred (list[list[[np.ndarray]]):
                pred (list) - list of images
                pred[i] (np.ndarray) - an image - a 2d np array of bounding boxes.
                    Shape nx6 where n is the number of bounding boxes in the image and 6 is the number of values in the bounding box: x1, y1, x2, y2, confidence, class of the bounding box
            gt (list[np.ndarray]) with shape list[nx6]: Same as above but for ground truth
            classes (list[int]): list of classes

        Returns:
            float: percent correctness
        """

        total_correct, total_localization, total_other, total_background, total_pred_bboxes = 0, 0, 0, 0, 0

        for image_idx in range(len(pred)):
            pred_image = pred[image_idx]
            gt_image = gt[image_idx]

            # no bounding box predictions in the image
            if len(pred_image) == 0:
                continue
            else:
                print("trace 0")
                correct, localization, other, background, total_pred_bboxes_in_image = (
                    self._get_yolo_metrics_from_boxes_in_image(pred_image, gt_image)
                )
                print("correct are", correct)
                total_correct += correct
                total_localization += localization
                total_other += other
                total_background += background
                total_pred_bboxes += total_pred_bboxes_in_image

        return (
            total_correct / total_pred_bboxes,
            total_localization / total_pred_bboxes,
            total_other / total_pred_bboxes,
            total_background / total_pred_bboxes,
            total_pred_bboxes,
        )

    def _get_yolo_metrics_from_boxes_in_image(
        self, pred_image: np.ndarray, gt_image: np.ndarray
    ) -> tuple[int, int, int, int, int]:
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
            ious = self._compute_iou(pred_bbox[np.newaxis, :4], gt_image[unmatched_gt_boxes_mask, :4])

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

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        """Computes the intersection over union between two twod arrays of bounding boxes

        Args:
            box1 (np.ndarray): nx4 bounding boxes. box1[i] is of the form [x1, y1, x2, y2]. if n is 1, then the single bounding box is compared to all bounding boxes in box2
            box2 (np.ndarray): nx4 bounding boxes. box1[i] is of the form [x1, y1, x2, y2]. if n is 1, then the single bounding box is compared to all bounding boxes in box1

        Returns:
            float: intersection over union
        """

        x1 = np.maximum(box1[:, 0], box2[:, 0])
        y1 = np.maximum(box1[:, 1], box2[:, 1])
        x2 = np.minimum(box1[:, 2], box2[:, 2])
        y2 = np.minimum(box1[:, 3], box2[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        union = area1 + area2 - intersection

        return intersection / union


class YOLOAccuracyMetric(YOLOMetric):
    def compute_value(self, pred: list[np.ndarray], gt: list[np.ndarray]) -> float:
        accuracy, _, _, _, _ = super().compute_values(pred, gt)

        return accuracy

    def get_name(self) -> str:
        return "YOLO Accuracy"

    def is_larger_better(self) -> bool:
        return True


class YOLOLocalizationMetric(YOLOMetric):
    def compute_value(self, pred: list[np.ndarray], gt: list[np.ndarray]) -> float:
        (_, localization, _, _, _) = super().compute_values(pred, gt)

        return localization

    def get_name(self) -> str:
        return "YOLO Localization"

    def is_larger_better(self) -> bool:
        return False


class YOLOOtherMetric(YOLOMetric):
    def compute_value(self, pred: list[np.ndarray], gt: list[np.ndarray]) -> float:
        (_, _, other, _, _) = super().compute_values(pred, gt)

        return other

    def get_name(self) -> str:
        return "YOLO Other"

    def is_larger_better(self) -> bool:
        return False


class YOLOBackgroundMetric(YOLOMetric):
    def compute_value(self, pred: list[np.ndarray], gt: list[np.ndarray]) -> float:
        (_, _, _, background, _) = super().compute_values(pred, gt)

        return background

    def get_name(self) -> str:
        return "YOLO Background"

    def is_larger_better(self) -> bool:
        return False
