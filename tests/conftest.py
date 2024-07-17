# test the iou function!!
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402


@pytest.fixture
def pred_gt_high_iou_boxes_with_high_confidence_and_class():
    """high IoU boxes with high confidence and class"""
    gt = np.array([105, 84, 798, 504, 3, 1])
    pred = np.array([110, 94, 798, 504, 3, 0.8])

    return pred, gt


@pytest.fixture
def pred_gt_medium_iou_boxes_with_high_confidence_and_class():
    """medium IoU boxes with high confidence and class"""
    # approx 25% iou
    gt = np.array([105, 84, 798, 504, 3, 1])
    pred = np.array([50, 50, 300, 504, 3, 0.8])

    return pred, gt


@pytest.fixture
def pred_gt_high_iou_boxes_with_wrong_class():
    """low IoU boxes with class and high IoU"""
    gt = np.array([105, 84, 798, 504, 3, 1])
    pred = np.array([105, 84, 798, 504, 4, 0.8])

    return pred, gt


@pytest.fixture
def pred_gt_identical_boxes():
    """identical boxes with full correctness"""
    gt = np.array([100, 200, 300, 400, 4, 1])
    pred = np.array([100, 200, 300, 400, 4, 0.6])

    return pred, gt


@pytest.fixture
def pred_gt_high_iou_boxes_correct_class_and_low_iou():
    """high iou box with correct class and low IoU"""
    gt = np.array([0, 0, 100, 100, 4, 1])
    pred = np.array([0, 0, 100, 80, 4, 0.2])

    return pred, gt


@pytest.fixture
def pred_gt_low_iou_box_correct_class_high_confidence():
    gt = np.array([0, 0, 100, 100, 5, 1])
    pred = np.array([80, 80, 100, 100, 5, 0.9])

    return pred, gt


@pytest.fixture
def pred_gt_low_iou_box_wrong_class_high_confidence():
    """low iou box with correct class and 0 IoU and wrong class"""

    gt = np.array([0, 0, 100, 100, 5, 1])
    pred = np.array([80, 80, 100, 100, 4, 0.9])

    return pred, gt


@pytest.fixture
def pred_gt_two_pred_boxes_close_to_gt_box_correct_order_of_confidence():
    """two pred boxes close to gt box, correct order of confidence"""
    gt = np.array([0, 0, 100, 100, 5, 1])
    pred = np.array([0, 0, 80, 80, 5, 0.9])
    pred2 = np.array([0, 0, 40, 40, 5, 0.8])

    return [pred, pred2], gt


@pytest.fixture
def pred_gt_two_pred_boxes_close_to_gt_box_wrong_order_of_confidence():
    """two pred boxes close to gt box, wrong order of confidence"""
    gt = np.array([0, 0, 100, 100, 5, 1])
    pred = np.array([0, 0, 40, 40, 5, 0.9])
    pred2 = np.array([0, 0, 80, 80, 5, 0.8])

    return [pred, pred2], gt


@pytest.fixture
def pred_gt_two_pred_boxes_close_to_gt_box_correct_order_of_confidence_wrong_class():
    """two pred boxes close to gt box, correct order of confidence, wrong class"""
    gt = np.array([0, 0, 100, 100, 5, 1])
    pred = np.array([0, 0, 80, 80, 4, 0.9])
    pred2 = np.array([0, 0, 40, 40, 5, 0.8])

    return [pred, pred2], gt


@pytest.fixture
def pred_gt_two_boxes_no_overlap_correct_class():
    """two boxes with no overlap, correct class"""
    gt = np.array([0, 0, 100, 100, 5, 1])
    pred = np.array([200, 200, 300, 300, 5, 0.8])

    return pred, gt


@pytest.fixture
def sample_unstable_precisions_and_recalls():
    precision = np.array([0.9, 0.8, 0.7, 0.2, 0.5, 0.4, 0.0, 0.2, 0.1, 0.0])
    recall = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    interp_precision = np.array([0.9, 0.8, 0.7, 0.5, 0.5, 0.4, 0.2, 0.2, 0.1, 0.0])
    interp_recall = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    return precision, recall, interp_precision, interp_recall


@pytest.fixture
def sample_stable_precisions_and_recalls():
    precision = np.array([0.9, 0.8, 0.7, 0.5, 0.5, 0.4, 0.2, 0.2, 0.1, 0.0])
    recall = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    return precision, recall, precision.copy(), recall.copy()
