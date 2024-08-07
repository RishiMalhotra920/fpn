from abc import abstractmethod

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from fpn.models.fast_rcnn_classifier import FastRCNNClassifier
from fpn.models.fpn import FPN
from fpn.models.model import Model
from fpn.models.rpn import RPN
from fpn.utils.batch_bounding_boxes import BatchBoundingBoxes


class FasterRCNNLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        p_i = x[0]
        p_i_star = x[1]

        # l_cls =
        pass


class FasterRCNN(Model):
    """this class takes in multiple cropped feature maps of arbitrary size, passes them to the RPN, gets the crops and objectness scores from
    the RPN, takes certain crops ta passes them to the FastRCNNClassifier, gets the softmax scores and bounding box regression offsets and

    In this experiment, B=256, C=3, H=224, W=224, nF=3,
    nBB=2000 (number of feature maps produced by FPN)
    nBB_pos = 500 (number of positive bounding boxes)
    nBB_neg = 1500 (number of negative bounding boxes)
    F=256 (feature map size) and R=256 (RoI)
    Compute the feature maps with FPN
        1. Incoming batch of images of shape (B, C, H, W)
        2. The batch is passed through the FPN to get three feature maps with the following ratios:
            [(B, F, H, W), (B, F, H/2, W/2), (B, F, H/4, W/4)]
        3. I freeze the pre-trained FPN.
    Pass the feature maps through the RPN
        3. We slide the RPN over each of [(B, F, H, W), (B, F, H/2, W/2), (B, F, H/4, W/4)] feature maps
           and produce [(B, H*W*nA), (B, H/2*W/2*nA), (B, H/4*W/4*nA)] cls scores and
            [(B, H*W*nA*4), (B, H/2*W/2*nA*4), (B, H/4*W/4*nA*4)] bboxes.
        4. We collapse the array and apply bbox regression offsets to the anchor boxes to get the bounding boxes in the image.
        5. Apply NMS here across the collated array per image and prepend batch_feature_map_index to the bounding box
            sample nBB_neg RoIs randomly where IoU(bbox, gt) < 0.3 (There is also a strategy for hard negative mining where we sample some negatives with high IoU to teach the model how to differentiate)
            sample nBB_pos RoIs where IoU(bbox, gt) > 0.5 to get a (num_rois, 5) num_rois <= nBB
            augmented_bbox[i] = np.ndarray([b_idx, x1, y1, x2, y2])

            3.1 we calculate the loss for the rpn:
                L_rpn_cls = cross_entropy(cls_i, cls_i_star)
                L_rpn_bbox = smooth_l1_loss(bbox_i, bbox_i_star)
                3.1.1
                    Translate the image space boxes to (B*nF, F, H, W) cls scores and (B*nF, F, nA*4) bbox regression offsets.
                    When translating bbox, find center of bbox and find the anchor box  aspect ratio that matches it most closely.
                    Then calculate feature_space_box_center/image_space_box_center to find which feature map pixel the box center is in.
                    Add the bbox regression offsets, t_x = (x - x_a) / w_a, t_y = (y - y_a) / h_a, t_w = log(w / w_a), t_h = log(h / h_a)
                    While you're here, also calculate the cli_s label for later use in 7.1.1
    Pass the cropped feature maps to the FastRCNNClassifier
        7. We pass the feature maps and augmented_bboxes to the FastRCNNClassifier
           We receive (num_rois, num_classes) softmax scores and (num_rois, 4) bounding box regression offsets.
            7.1 we calculate the loss for the fast_rcnn:
                L_fast_rcnn_cls = cross_entropy(cls_i, cls_i_star)
                L_fast_rcnn_bbox = smooth_l1_loss(bbox_i, bbox_i_star)
                Propagate the cls_i and bbox_i from 3.1.1 to here.
        8. Calculate bbox offsets compared to the original image.
            8.1 We use the bounding box regression offsets to adjust the bounding box coordinates output by the RPN.
        9. We pick the class with the highest softmax score and use the corresponding bounding box.
    Calculate the total loss
        11. The loss function is as follows:
            L_total = lambda_1 * L_rpn_cls + lambda_2 * L_rpn_bbox + lambda_3 * L_fast_rcnn_cls + lambda_4 * L_fast_rcnn_bbox


    Utility functions that I should code up:
        RPN output visualizer
        FastRCNN classifier output visualizer
        Full network output visualizer

    Other todos:
        Finetuning with optuna:
            Params to finetune: learning_rate, lambda_1, lambda_2, lambda_3, lambda_4, dropout_prob, nms_threshold
        Gradient clipping with max_norm to 1.0 to prevent exploding gradients early on
        Use warmup with cosine annealing learning rate scheduler.
        ---kaiming initialization--- (for another day :) )

    Args:
        Model (_type_): _description_
    """

    def __init__(self, nms_threshold: float):
        super().__init__(nms_threshold)
        self.backbone = FPN()
        self.fpn_map_small_anchor_scales = torch.tensor([32, 64, 128])
        self.fpn_map_medium_anchor_scales = torch.tensor([64, 128, 256])
        self.fpn_map_large_anchor_scales = torch.tensor([128, 256, 512])
        anchor_ratios = torch.tensor([0.5, 1, 2])
        self.all_anchor_scales = [
            self.fpn_map_small_anchor_scales,
            self.fpn_map_medium_anchor_scales,
            self.fpn_map_large_anchor_scales,
        ]
        self.all_anchor_ratios = [anchor_ratios, anchor_ratios, anchor_ratios]

        self.rpn = RPN(
            in_channels=256,
            num_anchor_scales=len(self.fpn_map_small_anchor_scales),
            num_anchor_ratios=len(anchor_ratios),
        )
        self.fast_rcnn_classifier = FastRCNNClassifier(num_classes=3)

        self.faster_rcnn_loss = FasterRCNNLoss()
        self.image_size = (224, 224)

    @abstractmethod
    def forward(self, x):
        fpn_maps = self.backbone(x)

        all_rpn_cls, all_rpn_rois, all_fast_rcnn_cls, all_fast_rcnn_bboxes = [], [], [], []
        for fpn_map, anchor_scale, anchor_ratio in zip(
            fpn_maps, self.all_anchor_scales, self.all_anchor_ratios
        ):  # (B, F, H/2, W/2), then (B, F, H/4, W/4), then (B, F, H/8, W/8)
            objectness, bbox_offset_volume = self.rpn(fpn_map)  # (b, s*s*9), (b, s*s*9, 4)

            bboxes = BatchBoundingBoxes.from_anchors_and_rpn_bbox_offset_volume(
                self.anchor_sizes, self.anchor_ratios, bbox_offset_volume, self.image_size
            )

            top_rpn_bboxes = BatchBoundingBoxes.pick_top_k_boxes_per_image(
                objectness.reshape(b, -1), bboxes, k=666, pos_to_neg_ratio=0.33
            )  # BoundingBoxes (b, nBB, 4jL)

            # pass to the fast rcnn classifier
            cls_probs, bbox_offsets_for_all_classes = self.fast_rcnn_classifier(
                fpn_maps, top_rpn_bboxes
            )  # (b, nBB, num_classes), (b, nBB, num_classes,4)

            # for each image, pick the class with the highest softmax score and use the corresponding bounding box
            cls, bbox_offsets = self._pick_top_class_and_bbox_offsets(cls_probs, bbox_offsets_for_all_classes)  # (b, nBB), (b, nBB, 4)

            fast_rcnn_bboxes = BatchBoundingBoxes.from_bounding_boxes_and_offsets(top_rpn_bboxes, bbox_offsets)

            all_rpn_cls.append(objectness)
            all_rpn_rois.append(top_rpn_bboxes)
            all_fast_rcnn_cls.append(cls)
            all_fast_rcnn_bboxes.append(fast_rcnn_bboxes)

        all_rpn_cls_tensor = torch.cat(all_rpn_cls, dim=1)
        all_rpn_rois_tensor = torch.cat(all_rpn_rois, dim=1)
        all_fast_rcnn_cls_tensor = torch.cat(all_fast_rcnn_cls, dim=1)
        all_fast_rcnn_bboxes_tensor = torch.cat(all_fast_rcnn_bboxes, dim=1)

        return all_rpn_cls_tensor, all_rpn_rois_tensor, all_fast_rcnn_cls_tensor, all_fast_rcnn_bboxes_tensor

        # 3.1 calculate the rpn loss
        # rpn_loss = FasterRCNNLoss(cls, bbox_offset_volume)

    def _pick_top_class_and_bbox_offsets(self, cls_probs, bbox_offsets_for_all_classes):
        b = cls_probs.shape[0]
        num_boxes = cls_probs.shape[1]
        batch_idx = torch.arange(b).unsqueeze(1).expand(b, num_boxes)  # (b, nBB) [[0, 1, 2, 3],[0, 1, 2, 3],[0, 1, 2, 3]]
        box_idx = torch.arange(num_boxes).unsqueeze(0).expand(b, num_boxes)  # (b, nBB) [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
        cls = cls_probs.argmax(dim=2)  # (b, nBB) eg: [[0, 2, 1, 3], [4, 4, 2, 3], [5, 1, 2, 3]]
        bbox_offsets = bbox_offsets_for_all_classes[batch_idx, box_idx, cls]  # (b, nBB, 4)
        return cls, bbox_offsets  # (b, nBB), (b, nBB, 4)

    @abstractmethod
    def predict(self, dataset: Dataset) -> list[np.ndarray]:
        """Given a dataset, predict the output of the model on the dataset.

        Args:
            dataset (Dataset): A dataset object.

        Returns:
            np.ndarray: A numpy array of predictions
                Shape: list[nx6]. ret[i] represents bounding boxes in image i.
                Shape of ret[i]: nx6 where n is the number of bounding boxes in image i
                and 6 is the number of values in the bounding box: x1, y1, x2, y2, confidence, class of the bounding
        """
        pass
