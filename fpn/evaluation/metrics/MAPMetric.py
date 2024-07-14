import numpy as np

from fpn.evaluation.metrics.metric import Metric


class MAPMetric(Metric):
    def __init__(self, iou_threshold: float):
        self.iou_threshold = iou_threshold
        self.CLASS_INDEX = 4
        self.CONF_INDEX = 5

    def compute_value(
        self, pred: list[list[np.ndarray]], gt: list[list[np.ndarray]]
    ) -> float:
        """Computes YOLO percent correctness according to the YOLOv1 paper.

        The classes are inferred from gt.

        Args:
            pred (list[list[[np.ndarray]]):
                pred (list) - list of images
                pred[i] (list) - an image represented as a list.
                pred[i][j] (np.ndarray) - a bounding box in the image
                    Shape nx6 where n is the number of bounding boxes in the image and 6 is the number of values in the bounding box: x1, y1, x2, y2, confidence, class of the
            gt (list[np.ndarray]) with shape list[nx6]: Same as above but for ground truth

        Returns:
            float: percent correctness
        """

        pred_image_indices = np.repeat(
            np.arange(len(pred)), [len(p) for p in pred]
        )  # num_boxes_across_all_images

        # gt - 10 images. len(gt) = 10
        #
        print("trace 10", np.arange(len(gt)), [len(g) for g in gt])
        gt_image_indices = np.repeat(np.arange(len(gt)), [len(g) for g in gt])

        collated_pred = np.vstack(pred)  # num_boxes_across_all_images x 6
        collated_gt = np.vstack(gt)  # num_boxes_across_all_images x 6

        print("collated_gt.shape", collated_gt.shape)
        print("gt_image_indices.shape", gt_image_indices.shape)

        classes = np.unique(collated_gt[:, self.CLASS_INDEX])
        print("this is classes", classes)
        average_precision_by_class = np.zeros_like(classes, dtype=float)

        for c_index, c in enumerate(classes):
            class_pred = collated_pred[collated_pred[:, self.CLASS_INDEX] == c]
            class_pred_image_indices = pred_image_indices[
                collated_pred[:, self.CLASS_INDEX] == c
            ]
            class_gt = collated_gt[collated_gt[:, self.CLASS_INDEX] == c]
            class_gt_image_indices = gt_image_indices[
                collated_gt[:, self.CLASS_INDEX] == c
            ]

            print("len gt", len(class_gt))
            if len(class_gt) == 0:
                average_precision_by_class[c_index] = 1.0
            elif len(class_pred) == 0:
                average_precision_by_class[c_index] = 0.0
            else:
                average_precision_by_class[c_index] = self._compute_average_precision(
                    class_pred,
                    class_gt,
                    class_pred_image_indices,
                    class_gt_image_indices,
                )

        return float(np.mean(average_precision_by_class))

    def _compute_average_precision(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        pred_image_indices: np.ndarray,
        gt_image_indices: np.ndarray,
    ) -> float:
        """Computes the average precision between all ground truth and predicted bounding boxes.

        We do this by sorting the predicted bounding boxes by confidence. We then go in descending order of confidence
        of predicted boxes, and for each predicted box, we calculate IoU with all ground truth boxes and assign the gt box
        with the highest IoU to the predicted box. We then assign that match to be a true positive if the IoU is above threshold
        and a false positive otherwise. We then calculate the precision and recall for all predictions up until that point. Then we calculate
        the average precision by interpolating the precision recall curve and calculating the area under the curve.

        Args:
            pred (np.ndarray): predicted bounding box values in the form [x1, y1, x2, y2, class, confidence]
                Shape (num_pred, 6)
            gt (np.ndarray): ground truth bounding box values in the form [x1, y1, x2, y2, class, confidence]
                Shape (num_gt, 6)
            pred_image_indices (np.ndarray): indices of the images for each prediction
                Shape (num_pred,)
            gt_image_indices (np.ndarray): indices of the images for each ground truth
                Shape (num_gt,)

        Returns:
            float: average precision
        """

        preds_order = pred[:, self.CONF_INDEX].argsort()[::-1]
        pred_sorted_by_conf = pred[preds_order]
        pred_image_indices = pred_image_indices[preds_order]

        precision = np.zeros(len(pred_sorted_by_conf))
        recall = np.zeros(len(pred_sorted_by_conf))
        tp = 0
        fp = 0
        gt_matched = np.full(len(gt), False)  # (num_gt,)
        gt_indices = np.arange(len(gt))  # (num_gt,)
        print("trace 1", gt)
        print("trace 2", pred_sorted_by_conf)
        for i, pred in enumerate(pred_sorted_by_conf):
            print("i is", i)
            pred_image_index = pred_image_indices[i]

            # boolean mask for gt boxes that are not matched and are in the same image as the prediction
            print(
                "shapes", gt_matched.shape, (gt_image_indices == pred_image_index).shape
            )
            unmatched_in_image_mask = (~gt_matched) & (
                gt_image_indices == pred_image_index
            )

            gt_unmatched_in_image_indices = gt_indices[unmatched_in_image_mask]
            gt_unmatched_in_image = gt[unmatched_in_image_mask]

            print("trace 5", gt_unmatched_in_image, pred)

            # TODO: rewrite for loop to use numpy ops and remove this for loop.
            # iou_scores = [
            #     self._compute_iou(pred[:4], gt_box[:4])
            #     for gt_box in gt_unmatched_in_image
            # ]

            # repeat the pred box len(gt_unmatched_in_image) times so that we can compute IoU with all gt boxes
            # TODO: may not even need to do the tile
            print("gt_unmatched_in_image.shape", gt_unmatched_in_image.shape)
            iou_scores = self._compute_iou(
                pred[:4][np.newaxis, :],
                gt_unmatched_in_image[:, :4],
            )

            print("this is iou_scores", iou_scores)

            max_iou_score_idx = np.argmax(iou_scores)
            iou = iou_scores[max_iou_score_idx]
            gt_match_idx = gt_unmatched_in_image_indices[max_iou_score_idx]
            gt_matched[gt_match_idx] = True
            print("trace 4", gt_match_idx, gt_matched)

            tp += 1 if iou > self.iou_threshold else 0
            fp += 0 if iou > self.iou_threshold else 1

            precision[i] = tp / (tp + fp)
            # len boxes is now length without the 0s.
            recall[i] = tp / len(gt)
            print("trace 3", gt_matched)

        precision, recall = self._interpolated_precision_recall(precision, recall)

        return np.trapz(np.concatenate([[1], precision]), np.concatenate([[0], recall]))

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        """Computes the intersection over union between two twod arrays of bounding boxes

        Args:
            box1 (np.ndarray): nx4 bounding boxes. box1[i] is of the form [x1, y1, x2, y2]. if n is 1, then the single bounding box is compared to all bounding boxes in box2
            box2 (np.ndarray): nx4 bounding boxes. box1[i] is of the form [x1, y1, x2, y2]. if n is 1, then the single bounding box is compared to all bounding boxes in box1

        Returns:
            float: intersection over union
        """
        print("this is box1", box1)
        print("this is box2", box2)

        x1 = np.maximum(box1[:, 0], box2[:, 0])
        y1 = np.maximum(box1[:, 1], box2[:, 1])
        x2 = np.minimum(box1[:, 2], box2[:, 2])
        y2 = np.minimum(box1[:, 3], box2[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        union = area1 + area2 - intersection

        print("this is x1", x1, y1, x2, y2)
        print("this is intersection", intersection, union, area1, area2)

        return intersection / union

    def _interpolated_precision_recall(
        self,
        precisions: np.ndarray,
        recalls: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolates precision values for each recall value.

        p_interp(r) = max(p(r')) for all r' >= r
        so if we have precision values [0.9, 0.8, 0.7, 0.2, 0.5, 0.4, 0.0, 0.2, 0.1] and recall values [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        interpolated_p = [0.9, 0.8, 0.7, 0.5, 0.5, 0.4, 0.2, 0.2, 0.1], this helps to smooth out the precision recall curve because
        there may be some inconsistencies in the precision recall curve. Smoothing makes this metric more stable.

        Args:
            precisions (np.ndarray): precision values
            recalls (np.ndarray): recall values

        Returns:
            tuple[np.ndarray, np.ndarray]: interpolated recall and precision values
        """
        order = np.argsort(recalls)
        recalls = recalls[order]
        precisions = precisions[order]

        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])

        return precisions, recalls

    def get_name(self) -> str:
        return f"MAP@{self.iou_threshold}"

    def is_larger_better(self) -> bool:
        return True
