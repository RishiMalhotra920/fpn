import numpy as np

from fpn.evaluation.metrics.YOLO_metrics import (
    YOLOAccuracyMetric,
    YOLOBackgroundMetric,
    YOLOLocalizationMetric,
    YOLOOtherMetric,
)
from fpn.utils.compute_iou import compute_iou


# test for two images three boxes each - 100% correctness
def test_metrics_identical_boxes_with_full_correctness(
    pred_gt_high_iou_boxes_with_high_confidence_and_class,
):
    pred_one = pred_gt_high_iou_boxes_with_high_confidence_and_class[1]

    # 2 images, 3 boxes each.
    pred_image = np.vstack([pred_one, pred_one, pred_one])
    gt_image = np.vstack([pred_one, pred_one, pred_one])

    pred_images = [pred_image, pred_image]
    gt_images = [gt_image, gt_image]

    accuracy_metric = YOLOAccuracyMetric()
    localization_metric = YOLOLocalizationMetric()
    other_metric = YOLOOtherMetric()
    background_metric = YOLOBackgroundMetric()

    assert accuracy_metric.compute_value(pred_images, gt_images) == 1.0, "Accuracy metric failed"
    assert localization_metric.compute_value(pred_images, gt_images) == 0.0, "Localization metric failed"
    assert other_metric.compute_value(pred_images, gt_images) == 0.0, "Other metric failed"
    assert background_metric.compute_value(pred_images, gt_images) == 0.0, "Background metric failed"


def test_MAP_metric_boxes_in_background_class(
    pred_gt_two_boxes_no_overlap_correct_class,
):
    pred_one, gt_one = pred_gt_two_boxes_no_overlap_correct_class

    # 2 images, 3 boxes each.
    pred_image = np.vstack([pred_one, pred_one, pred_one])
    gt_image = np.vstack([gt_one, gt_one, gt_one])
    pred = [pred_image, pred_image]
    gt = [gt_image, gt_image]

    accuracy_metric = YOLOAccuracyMetric()
    localization_metric = YOLOLocalizationMetric()
    other_metric = YOLOOtherMetric()
    background_metric = YOLOBackgroundMetric()

    assert accuracy_metric.compute_value(pred, gt) == 0.0, "Accuracy metric failed"
    assert localization_metric.compute_value(pred, gt) == 0.0, "Localization metric failed"
    assert other_metric.compute_value(pred, gt) == 0.0, "Other metric failed"
    assert background_metric.compute_value(pred, gt) == 1.0, "Background metric failed"


def test_MAP_metric_num_pred_neq_num_gt(
    pred_gt_high_iou_boxes_with_high_confidence_and_class,
):
    """one image has three predictions, one has two. one image has zero predictions. one image has same number of preds as gt"""
    pred, gt = pred_gt_high_iou_boxes_with_high_confidence_and_class

    # TP: 2, FP: 1.
    pred_image_1 = np.array([pred for _ in range(3)])
    gt_image_1 = np.array([gt for _ in range(2)])

    # TP: 0, FP: 0.
    pred_image_2 = np.array([])
    gt_image_2 = np.array([gt for _ in range(3)])

    # TP: 3, FP: 0
    pred_image_3 = np.array([pred for _ in range(3)])
    gt_image_3 = np.array([gt for _ in range(3)])

    pred = [pred_image_1, pred_image_2, pred_image_3]
    gt = [gt_image_1, gt_image_2, gt_image_3]

    accuracy_metric = YOLOAccuracyMetric()
    localization_metric = YOLOLocalizationMetric()
    other_metric = YOLOOtherMetric()
    background_metric = YOLOBackgroundMetric()

    assert accuracy_metric.compute_value(pred, gt) == 5 / 6, "Accuracy metric failed"
    assert localization_metric.compute_value(pred, gt) == 0.0, "Localization metric failed"
    assert other_metric.compute_value(pred, gt) == 0.0, "Other metric failed"
    assert background_metric.compute_value(pred, gt) == 1 / 6, "Background metric failed"


def test_metrics_background_error(
    pred_gt_low_iou_box_wrong_class_high_confidence, pred_gt_two_boxes_no_overlap_correct_class
):
    pred, gt = pred_gt_low_iou_box_wrong_class_high_confidence  # background. iou = 0.04
    pred2, gt2 = pred_gt_two_boxes_no_overlap_correct_class  # background

    # 5 images, 2 boxes each.
    pred_image = np.vstack([pred, pred2])
    gt_image = np.vstack([gt, gt2])
    pred_images = [pred_image for _ in range(5)]
    gt_images = [gt_image for _ in range(5)]

    accuracy_metric = YOLOAccuracyMetric()
    localization_metric = YOLOLocalizationMetric()
    other_metric = YOLOOtherMetric()
    background_metric = YOLOBackgroundMetric()

    assert accuracy_metric.compute_value(pred_images, gt_images) == 0.0, "Accuracy metric failed"
    assert localization_metric.compute_value(pred_images, gt_images) == 0.0, "Localization metric failed"
    assert other_metric.compute_value(pred_images, gt_images) == 0.0, "Other metric failed"
    assert background_metric.compute_value(pred_images, gt_images) == 1.0, "Background metric failed"


def test_metrics_localization_error(pred_gt_medium_iou_boxes_with_high_confidence_and_class):
    pred, gt = pred_gt_medium_iou_boxes_with_high_confidence_and_class  # background. iou = 0.04

    # 5 images, 2 boxes each.
    pred_image = np.vstack([pred])
    gt_image = np.vstack([gt])
    pred_images = [pred_image for _ in range(5)]
    gt_images = [gt_image for _ in range(5)]

    accuracy_metric = YOLOAccuracyMetric()
    localization_metric = YOLOLocalizationMetric()
    other_metric = YOLOOtherMetric()
    background_metric = YOLOBackgroundMetric()

    assert accuracy_metric.compute_value(pred_images, gt_images) == 0.0, "Accuracy metric failed"
    assert localization_metric.compute_value(pred_images, gt_images) == 1.0, "Localization metric failed"
    assert other_metric.compute_value(pred_images, gt_images) == 0.0, "Other metric failed"
    assert background_metric.compute_value(pred_images, gt_images) == 0.0, "Background metric failed"


def test_metrics_other_error(pred_gt_high_iou_boxes_with_wrong_class):
    pred, gt = pred_gt_high_iou_boxes_with_wrong_class  # background. iou = 0.04

    # 5 images, 2 boxes each.
    pred_image = np.vstack([pred])
    gt_image = np.vstack([gt])
    pred_images = [pred_image for _ in range(5)]
    gt_images = [gt_image for _ in range(5)]

    accuracy_metric = YOLOAccuracyMetric()
    localization_metric = YOLOLocalizationMetric()
    other_metric = YOLOOtherMetric()
    background_metric = YOLOBackgroundMetric()

    assert accuracy_metric.compute_value(pred_images, gt_images) == 0.0, "Accuracy metric failed"
    assert localization_metric.compute_value(pred_images, gt_images) == 0.0, "Localization metric failed"
    assert other_metric.compute_value(pred_images, gt_images) == 1.0, "Other metric failed"
    assert background_metric.compute_value(pred_images, gt_images) == 0.0, "Background metric failed"


# test iou function with two boxes with 0 overlap
def test_compute_iou_no_overlap_and_full_overlap(
    pred_gt_two_boxes_no_overlap_correct_class,
):
    pred, gt = pred_gt_two_boxes_no_overlap_correct_class
    pred, gt = pred[:4], gt[:4]

    metric = YOLOAccuracyMetric()
    assert np.all(compute_iou(np.array([pred]), np.array([gt, gt, gt])) == np.array([0.0, 0.0, 0.0]))
    assert np.all(compute_iou(np.array([pred]), np.array([gt])) == np.array([0.0]))
    assert np.all(compute_iou(np.array([pred, pred]), np.array([gt, gt])) == np.array([0.0, 0.0]))
    assert np.all(compute_iou(np.array([pred, pred]), np.array([gt])) == np.array([0.0, 0.0]))

    assert np.all(compute_iou(np.array([pred, pred, pred]), np.array([pred, pred, pred])) == np.array([1.0, 1.0, 1.0]))


def test_compute_iou_mixed_overlaps(
    pred_gt_two_boxes_no_overlap_correct_class,
    pred_gt_two_pred_boxes_close_to_gt_box_wrong_order_of_confidence,
):
    pred1, gt1 = pred_gt_two_boxes_no_overlap_correct_class
    (pred2, pred3), gt2 = pred_gt_two_pred_boxes_close_to_gt_box_wrong_order_of_confidence
    pred1, gt1 = pred1[:4], gt1[:4]
    pred2, pred3, gt2 = pred2[:4], pred3[:4], gt2[:4]

    metric = YOLOAccuracyMetric()

    assert np.all(
        compute_iou(np.array([pred1, pred2, pred3]), np.array([gt1, gt2, gt2]))
        == np.array([0.0, (40 * 40) / (100 * 100), (80 * 80) / (100 * 100)])
    )
