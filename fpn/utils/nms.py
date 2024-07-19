import numpy as np

from fpn.utils.compute_iou import compute_iou


def apply_nms_by_class(pred: list[np.ndarray], iou_threshold: float) -> list[np.ndarray]:
    """Applies non-max suppression to the predictions by class.

    For any two boxes of the same class, if the IOU between them is greater than iou_threshold, the box with the lower confidence is suppressed.

    Args:
        preds: np.ndarray of shape (num_boxes, 6) where each row is [x1, y1, x2, y2, confidence, class]
        iou_threshold: float, threshold for non-max suppression

    Returns:
        list[np.ndarray]: list of np.ndarrays of shape (num_boxes, 6) where each row is [x1, y1, x2, y2, confidence, class].
            Represents the boxes that are not suppressed in each image.
    """

    nms_pred = []
    for image_pred in pred:
        if len(image_pred) == 0:  # no bounding box predictions in the image
            nms_pred.append(image_pred)
            continue

        print("trace 0", image_pred[:, 5])
        classes = np.unique(image_pred[:, 5])
        indices = np.arange(len(image_pred))

        mask = np.ones(len(image_pred), dtype=bool)  # (n,) boxes to keep for this image.
        for c in range(len(classes)):
            class_mask = image_pred[:, 5] == classes[c]  # (n,)
            class_pred = image_pred[class_mask]  # (m, 6) where m is the number of boxes of class c
            class_indices = indices[class_mask]  # (m,)

            order = np.argsort(class_pred[:, 4])[::-1]
            class_pred = class_pred[order]
            class_indices = class_indices[order]

            for i in range(len(class_pred)):
                print("c", c, "i", i, "class_pred", class_pred, "class_indices", class_indices, "mask", mask)
                if not mask[class_indices[i]]:
                    continue

                iou = compute_iou(class_pred[i, :4].reshape(1, 4), class_pred[i + 1 :, :4])
                print("iou", iou)
                print("------")
                mask[class_indices[i + 1 :]] = mask[class_indices[i + 1 :]] & (iou < iou_threshold)

        nms_pred.append(image_pred[mask])
    return nms_pred
