import numpy as np


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
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
