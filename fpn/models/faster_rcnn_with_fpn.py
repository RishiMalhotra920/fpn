import torch
from torch import nn

from fpn.models.fpn import FPN

from .faster_rcnn import FasterRCNN


class FasterRCNNWithFPN(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        nms_threshold: float,
        num_rpn_rois_to_sample: int = 2000,
        rpn_pos_to_neg_ratio: float = 0.33,
        rpn_pos_iou: float = 0.7,
        rpn_neg_iou: float = 0.3,
    ):
        super().__init__()
        self.backbone = FPN()
        self.fpn_map_small_anchor_scales = torch.tensor([32.0, 64.0, 128.0])
        self.fpn_map_medium_anchor_scales = torch.tensor([64.0, 128.0, 256.0])
        self.fpn_map_large_anchor_scales = torch.tensor([128.0, 256.0, 512.0])
        anchor_ratios = torch.tensor([0.5, 1, 2])
        self.all_anchor_scales = [
            self.fpn_map_small_anchor_scales,
            self.fpn_map_medium_anchor_scales,
            self.fpn_map_large_anchor_scales,
        ]
        self.all_anchor_ratios = [anchor_ratios, anchor_ratios, anchor_ratios]

        self.all_anchor_widths = []  # list([(9, ), (9, ), (9, )])
        self.all_anchor_heights = []  # list([(9, ), (9, ), (9, )])

        for anchor_scales in self.all_anchor_scales:
            permutations = torch.cartesian_prod(anchor_scales, anchor_ratios)
            widths = permutations[:, 0] * permutations[:, 1]  # (9, )
            heights = permutations[:, 0] * (1 / permutations[:, 1])  # (9, )
            self.all_anchor_widths.append(widths)
            self.all_anchor_heights.append(heights)

        self.image_size = image_size
        self.nms_threshold = nms_threshold
        self.num_rpn_rois_to_sample = num_rpn_rois_to_sample
        self.rpn_pos_to_neg_ratio = rpn_pos_to_neg_ratio
        self.rpn_pos_iou = rpn_pos_iou
        self.rpn_neg_iou = rpn_neg_iou
        self.faster_rcnn = FasterRCNN(
            image_size=image_size,
            nms_threshold=nms_threshold,
            num_rpn_rois_to_sample=num_rpn_rois_to_sample,
            rpn_pos_to_neg_ratio=rpn_pos_to_neg_ratio,
            rpn_pos_iou=rpn_pos_iou,
            rpn_neg_iou=rpn_neg_iou,
        )

    def __call__(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        """Forward pass of the model.

        Passes the image through the FPN backbone, then the RPN, then the FastRCNNClassifier.

        Args:
            x (torch.Tensor): input image of shape (B, 3, H, W)

        Returns:
        """
        fpn_maps = self.backbone(x)  # tuple[(N, 256, M/4, M/4), (N, 256, M/8, M/8), (N, 256, M/16, M/16)]
        # fpn_map shape: (b, f, s, s), anchor_heights: (9, ), anchor_widths: (9, )
        rpn_objectness_list = []
        rpn_bboxes_list = []
        fast_rcnn_cls_list = []
        fast_rcnn_bboxes_list = []
        for fpn_map, anchor_heights, anchor_widths in zip(fpn_maps, self.all_anchor_heights, self.all_anchor_widths):
            rpn_objectness, rpn_bboxes, fast_rcnn_cls, list_of_bboxes_with_cls = self.faster_rcnn(fpn_map, anchor_heights, anchor_widths)
            # not the most efficient, eg: i would like to do a nms across fpn maps after the RPN stage
            # but time is of the essence here.
            rpn_objectness_list.append(rpn_objectness)
            rpn_bboxes_list.append(rpn_bboxes)
            fast_rcnn_cls_list.append(fast_rcnn_cls)
            fast_rcnn_bboxes_list.append(list_of_bboxes_with_cls)

        return rpn_objectness_list, rpn_bboxes_list, fast_rcnn_cls_list, fast_rcnn_bboxes_list
