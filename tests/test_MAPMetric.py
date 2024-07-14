import numpy as np
from fpn.evaluation.metrics.MAPMetric import MAPMetric


# test for two images three boxes each - 100% correctness
def test_MAP_metric_identical_boxes_with_full_correctness(
    pred_gt_high_iou_boxes_with_high_confidence_and_class,
):
    pred_one = pred_gt_high_iou_boxes_with_high_confidence_and_class[1]

    # 2 images, 3 boxes each.
    gt_image = [pred_one for _ in range(3)]
    gt_image_2 = [pred_one for _ in range(3)]
    gt = [gt_image, gt_image_2]
    pred = gt.copy()
    metric = MAPMetric(0.5)

    assert metric.compute_value(pred, gt) == 1.0


def test_MAP_metric_boxes_with_zero_correctness(
    pred_gt_two_boxes_no_overlap_correct_class,
):
    pred_one, gt_one = pred_gt_two_boxes_no_overlap_correct_class

    # 2 images, 3 boxes each.
    pred_image = [pred_one for _ in range(3)]
    gt_image = [gt_one for _ in range(3)]
    pred = [pred_image, pred_image]
    gt = [gt_image, gt_image]
    metric = MAPMetric(0.5)

    assert metric.compute_value(pred, gt) == 0.0


def test_MAP_metric_num_pred_neq_num_gt(
    pred_gt_high_iou_boxes_with_high_confidence_and_class,
):
    """one image has three predictions, one has two. one image has zero predictions. one image has same number of preds as gt"""
    pred, gt = pred_gt_high_iou_boxes_with_high_confidence_and_class

    # TP: 2, FP: 1.
    pred_image_1 = [pred for _ in range(3)]
    gt_image_1 = [gt for _ in range(2)]

    # TP: 0, FP: 0.
    pred_image_2 = []
    gt_image_2 = [gt for _ in range(3)]

    # TP: 3, FP: 0
    pred_image_3 = [pred for _ in range(3)]
    gt_image_3 = [gt for _ in range(3)]

    pred = [pred_image_1, pred_image_2, pred_image_3]
    gt = [gt_image_1, gt_image_2, gt_image_3]
    metric = MAPMetric(0.5)

    # precision per class is [5/6]

    assert metric.compute_value(pred, gt) == 5 / 6


# more predictions than labels per image test

# test for five images, two boxes each. prediction has bounding boxes correct, but classes all wrong.
# test for five images, two boxes each. prediction has zero boxes.

# test for two images, four boxes each - different confidence values. checking if sorting works correctly: 100% correctness


# test iou function with two boxes with 0 overlap
def test_compute_iou_no_overlap_and_full_overlap(
    pred_gt_two_boxes_no_overlap_correct_class,
):
    pred, gt = pred_gt_two_boxes_no_overlap_correct_class
    pred, gt = pred[:4], gt[:4]

    metric = MAPMetric(0.5)
    assert np.all(
        metric._compute_iou(np.array([pred]), np.array([gt, gt, gt]))
        == np.array([0.0, 0.0, 0.0])
    )

    assert np.all(
        metric._compute_iou(np.array([pred]), np.array([gt])) == np.array([0.0])
    )
    assert np.all(
        metric._compute_iou(np.array([pred, pred]), np.array([gt, gt]))
        == np.array([0.0, 0.0])
    )
    assert np.all(
        metric._compute_iou(np.array([pred, pred]), np.array([gt]))
        == np.array([0.0, 0.0])
    )

    assert np.all(
        metric._compute_iou(np.array([pred, pred, pred]), np.array([pred, pred, pred]))
        == np.array([1.0, 1.0, 1.0])
    )


def test_compute_iou_mixed_overlaps(
    pred_gt_two_boxes_no_overlap_correct_class,
    pred_gt_two_pred_boxes_close_to_gt_box_wrong_order_of_confidence,
):
    pred1, gt1 = pred_gt_two_boxes_no_overlap_correct_class
    (pred2, pred3), gt2 = (
        pred_gt_two_pred_boxes_close_to_gt_box_wrong_order_of_confidence
    )
    pred1, gt1 = pred1[:4], gt1[:4]
    pred2, pred3, gt2 = pred2[:4], pred3[:4], gt2[:4]

    metric = MAPMetric(0.5)

    assert np.all(
        metric._compute_iou(np.array([pred1, pred2, pred3]), np.array([gt1, gt2, gt2]))
        == np.array([0.0, (40 * 40) / (100 * 100), (80 * 80) / (100 * 100)])
    )


def test_unstable_interpolated_precision_recall(sample_unstable_precisions_and_recalls):
    precision, recall, interp_precision, interp_recall = (
        sample_unstable_precisions_and_recalls
    )
    metric = MAPMetric(0.5)
    interpolated_precision, interpolated_recall = metric._interpolated_precision_recall(
        precision, recall
    )

    assert np.all(interpolated_precision == interp_precision)
    assert np.all(interpolated_recall == interp_recall)


def test_stable_interpolated_precision_recall(sample_stable_precisions_and_recalls):
    precision, recall, interp_precision, interp_recall = (
        sample_stable_precisions_and_recalls
    )
    metric = MAPMetric(0.5)
    interpolated_precision, interpolated_recall = metric._interpolated_precision_recall(
        precision, recall
    )

    assert np.all(interpolated_precision == interp_precision)
    assert np.all(interpolated_recall == interp_recall)
