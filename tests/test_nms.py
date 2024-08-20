import numpy as np

from fpn.utils.nms import apply_nms_by_class  # Import your function


def test_nms_by_class_single_class():
    image_with_boxes = np.array([[0, 0, 10, 10, 0.9, 0], [1, 1, 11, 11, 0.8, 0], [20, 20, 30, 30, 0.7, 0]])

    pred = [image_with_boxes]

    nms_pred = apply_nms_by_class(pred, 0.5)
    print("nms_pred", nms_pred)
    assert np.allclose(nms_pred, np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 30, 30, 0.7, 0]]))


def test_nms_by_class_multiple_classes():
    image = np.array(
        [
            [0, 0, 10, 10, 0.9, 0],
            [1, 1, 11, 11, 0.8, 0],
            [20, 20, 30, 30, 0.7, 0],
            [0, 0, 10, 10, 0.9, 1],
            [21, 21, 31, 31, 0.8, 1],
        ]
    )
    pred = [image]

    # classes
    nms_pred = apply_nms_by_class(pred, iou_threshold=0.5)

    expected_bbox = [
        np.array(
            [
                [0, 0, 10, 10, 0.9, 0],  # Highest score for class 0
                [20, 20, 30, 30, 0.7, 0],  # Non-overlapping box for class 0
                [0, 0, 10, 10, 0.9, 1],  # Highest score for class 1
                [21, 21, 31, 31, 0.8, 1],  # Non-overlapping box for class 1
            ]
        )
    ]

    assert np.allclose(nms_pred, expected_bbox)


def test_nms_by_class_empty_input():
    pred = [np.array([]), np.array([]), np.array([])]

    nms_pred = apply_nms_by_class(pred, iou_threshold=0.5)
    assert np.allclose(nms_pred, [np.array([]), np.array([]), np.array([])])


def test_nms_by_class_no_suppressions():
    image = np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 30, 30, 0.8, 1], [40, 40, 50, 50, 0.7, 2]])

    pred = [image]

    nms_pred = apply_nms_by_class(pred, iou_threshold=0.5)
    assert np.allclose(nms_pred, pred)


def test_nms_by_class_mixed_suppressions():
    image = np.array(
        [
            [0, 0, 10, 10, 0.9, 0],
            [1, 1, 11, 11, 0.8, 0],
            [20, 20, 30, 30, 0.7, 0],
            [0, 0, 10, 10, 0.9, 1],
            [21, 21, 31, 31, 0.8, 1],
            [40, 40, 50, 50, 0.7, 2],
        ]
    )

    pred = [image]

    nms_pred = apply_nms_by_class(pred, iou_threshold=0.5)

    expected_nms_pred = np.array(
        [
            [0, 0, 10, 10, 0.9, 0],
            [20, 20, 30, 30, 0.7, 0],
            [0, 0, 10, 10, 0.9, 1],
            [21, 21, 31, 31, 0.8, 1],
            [40, 40, 50, 50, 0.7, 2],
        ],
        dtype=np.float32,
    )

    assert np.allclose(nms_pred, expected_nms_pred)
