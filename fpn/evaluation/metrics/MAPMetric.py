import numpy as np

from fpn.evaluation.metrics import Metric


class MAPMetric(Metric):
    def __init__(self, iou_threshold: float):
        self.iou_threshold = iou_threshold
        self.CLASS_INDEX = 4
        self.CONF_INDEX = 5
        self.IMAGE_INDEX = 6

    def compute_value(
        self, gt: list[np.ndarray], pred: list[np.ndarray], num_classes: int
    ) -> float:
        """Computes YOLO percent correctness according to the YOLOv1 paper.

        Args:
            gt (list[np.ndarray]) with shape list[nx6]: gt[i] contains bounding box, confidence, class for image i in the form: [x1, y1, x2, y2, confidence, class]
            pred (list[np.ndarray]) with shape list[nx6]: pred[i] contains bounding box, confidence, class for image i in the form: [x1, y1, x2, y2, confidence, class]
            num_classes (int): number of classes

        Returns:
            float: percent correctness
        """

        collated_gt_arr = []
        collated_pred_arr = []

        for i in range(len(gt)):
            collated_gt_arr.append(np.concatenate(gt[i], i))

        for i in range(len(pred)):
            collated_pred_arr.append(np.concatenate(pred[i], i))

        collated_gt = np.array(
            collated_gt_arr
        )  # shape: nx7 num_boxes_across_all_images x [x1, y1, x2, y2, confidence, class, image_index]
        collated_pred = np.array(
            collated_pred_arr
        )  # shape: nx7 num_boxes_across_all_images x [x1, y1, x2, y2, confidence, class, image_index]

        average_precision_by_class = np.zeros(num_classes)

        for c in range(num_classes):
            class_pred = collated_pred[collated_pred[:, self.CLASS_INDEX] == c]
            class_gt = collated_gt[collated_gt[:, self.CLASS_INDEX] == c]

            if len(class_gt) == 0:
                average_precision_by_class[c] = 1.0
            elif len(class_pred) == 0:
                average_precision_by_class[c] = 0.0
            else:
                average_precision_by_class[c] = self.compute_average_precision(
                    class_gt, class_pred
                )

        return float(np.mean(average_precision_by_class))

    def compute_average_precision(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Computes the average precision between all ground truth and predicted bounding boxes.

        We do this by sorting the predicted bounding boxes by confidence. We then go in descending order of confidence
        of predicted boxes, and for each predicted box, we calculate IoU with all ground truth boxes and assign the gt box
        with the highest IoU to the predicted box. We then assign that match to be a true positive if the IoU is above threshold
        and a false positive otherwise. We then calculate the precision and recall for all predictions up until that point. Then we calculate
        the average precision by interpolating the precision recall curve and calculating the area under the curve.

        Args:
            gt (np.ndarray): ground truth bounding box values in the form [x1, y1, x2, y2, class, confidence]
            pred (np.ndarray): predicted bounding box values in the form [x1, y1, x2, y2, class, confidence]

        Returns:
            float: average precision
        """

        pred_sorted_by_conf = pred[pred[:, self.CONF_INDEX].argsort()[::-1]]
        # gt_boxes = gt[:, :4]
        precision = np.zeros(len(pred_sorted_by_conf))
        recall = np.zeros(len(pred_sorted_by_conf))

        tp = 0
        fp = 0

        gt_matched = np.full(len(gt), False)
        for i, pred in enumerate(pred_sorted_by_conf):
            pred_box = pred[:4]
            pred_image_index = pred[self.IMAGE_INDEX]
            gt_in_image = gt[gt[:, self.IMAGE_INDEX] == pred_image_index]
            gt_unmatched_boxes_in_image = gt_in_image[~gt_matched, :4]

            iou_scores = [
                self.compute_iou(pred_box, gt_box)
                for gt_box in gt_unmatched_boxes_in_image
            ]

            gt_match_idx = np.argmax(iou_scores)
            iou = iou_scores[gt_match_idx]
            gt_matched[gt_match_idx] = True

            tp += 1 if iou > self.iou_threshold else 0
            fp += 0 if iou > self.iou_threshold else 1

            precision[i] = tp / (tp + fp)
            # len boxes is now length without the 0s.
            recall[i] = tp / len(gt)

        precision, recall = self.interpolated_precision_recall(recall, precision)
        return np.trapz(precision, recall)

    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Computes the intersection over union between two bounding boxes.

        Args:
            box1 (np.ndarray): bounding box values in the form [x1, y1, x2, y2]
            box2 (np.ndarray): bounding box values in the form [x1, y1, x2, y2]

        Returns:
            float: intersection over union
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union

    def interpolated_precision_recall(
        self, recalls: np.ndarray, precisions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolates precision values for each recall value.

        Args:
            recalls (np.ndarray): recall values
            precisions (np.ndarray): precision values

        Returns:
            tuple[np.ndarray, np.ndarray]: interpolated recall and precision values
        """
        order = np.argsort(recalls)
        recalls = recalls[order]
        precisions = precisions[order]

        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])

        return recalls, precisions

    def get_name(self) -> str:
        return f"MAP@{self.iou_threshold}"

    def is_larger_better(self) -> bool:
        return True
