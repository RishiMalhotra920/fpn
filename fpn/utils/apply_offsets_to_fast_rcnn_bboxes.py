import torch


def apply_offsets_to_fast_rcnn_bboxes(self, list_of_picked_bboxes: list[torch.Tensor], offsets: list[torch.Tensor]) -> list[torch.Tensor]:
    """Apply the bounding box regression offsets to the bounding boxes.

    Args:
        list_of_picked_bboxes (list[b, torch.Tensor(L_i, 4)]): list of bounding boxes in corner format of shape
        offsets (list[b, torch.Tensor(L_i, 4)]): list of bounding box offsets

    Returns:
        list[b, torch.Tensor(L_i, 4)]: bounding boxes in corner format
    """

    list_of_bboxes_with_offsets = []
    for image_bboxes, image_bbox_offsets in zip(list_of_picked_bboxes, offsets):
        image_bboxes_with_offsets = torch.zeros_like(image_bboxes)

        prev_bboxes_width = image_bboxes[:, 2] - image_bboxes[:, 0]
        prev_bboxes_height = image_bboxes[:, 3] - image_bboxes[:, 1]

        image_bboxes_with_offsets[:, 0] = image_bbox_offsets[:, 0] * prev_bboxes_width + image_bboxes[:, 0]
        image_bboxes_with_offsets[:, 1] = image_bbox_offsets[:, 1] * prev_bboxes_height + image_bboxes[:, 1]
        image_bboxes_with_offsets[:, 2] = image_bboxes_with_offsets[:, 0] + torch.exp(image_bbox_offsets[:, 2]) * prev_bboxes_width
        image_bboxes_with_offsets[:, 3] = image_bboxes_with_offsets[:, 1] + torch.exp(image_bbox_offsets[:, 3]) * prev_bboxes_height

        list_of_bboxes_with_offsets.append(image_bboxes_with_offsets)

    return list(list_of_bboxes_with_offsets)
