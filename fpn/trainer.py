from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fpn.data.VOC_data import CustomVOCDetectionDataset
from fpn.loss.faster_rcnn_loss import FasterRCNNLoss
from fpn.models.faster_rcnn import FasterRCNN
from fpn.models.fpn import FPN
from fpn.run_manager import RunManager
from fpn.YOLO_metrics import YOLOMetrics


def log_gradients(model: nn.Module) -> None:
    """
    TODO: log gradients here
    """
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         avg_grad = param.grad.abs().mean().item()
    #         avg_weight = param.abs().mean().item()
    #         avg_grad_weight = avg_grad / avg_weight
    #         print(
    #             f"Layer: {name} | Avg Grad: {avg_grad} | Avg Weight: {avg_weight} | Avg Grad/Weight: {avg_grad_weight}"
    #         )
    pass


class Trainer:
    def __init__(
        self,
        backbone: FPN,
        model: FasterRCNN,  # pass in DataParallel here but leave this for type hinting
        train_dataloader: DataLoader[CustomVOCDetectionDataset],
        val_dataloader: DataLoader,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        optimizer: torch.optim.Optimizer,
        loss_fn: FasterRCNNLoss,
        metric: YOLOMetrics,
        epoch_start: int,
        epoch_end: int,
        run_manager: RunManager,
        checkpoint_interval: int,
        log_interval: int,
        *,
        device: str,
        image_size: tuple[int, int],
        nms_threshold: float,
        num_rpn_rois_to_sample: int = 2000,
        rpn_pos_to_neg_ratio: float = 0.33,
        rpn_pos_iou: float = 0.7,
        rpn_neg_iou: float = 0.3,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.run_manager = run_manager
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval
        self.device = device
        self.metric = metric
        self.backbone = backbone

        # fpn specific things
        self.fpn_map_small_anchor_scales = torch.tensor([32.0, 64.0, 128.0], device=device)
        self.fpn_map_medium_anchor_scales = torch.tensor([64.0, 128.0, 256.0], device=device)
        self.fpn_map_large_anchor_scales = torch.tensor([128.0, 256.0, 512.0], device=device)
        anchor_ratios = torch.tensor([0.5, 1, 2], device=device)
        self.all_anchor_scales = [
            self.fpn_map_small_anchor_scales,
            self.fpn_map_medium_anchor_scales,
            self.fpn_map_large_anchor_scales,
        ]

        self.all_anchor_ratios = [anchor_ratios, anchor_ratios, anchor_ratios]

        self.all_anchor_widths = []  # list([(9, ), (9, ), (9, )])
        self.all_anchor_heights = []  # list([(9, ), (9, ), (9, )])
        self.all_anchor_positions = []
        feature_map_dims = [28, 14, 7]  # found through experimentation
        for anchor_scales, s in zip(self.all_anchor_scales, feature_map_dims):
            permutations = torch.cartesian_prod(anchor_scales, anchor_ratios)
            widths = permutations[:, 0] * permutations[:, 1]  # (9, )
            heights = permutations[:, 0] * (1 / permutations[:, 1])  # (9, )

            anchor_positions = self._get_anchor_positions(widths, heights, s, image_size[0])

            self.all_anchor_widths.append(widths)
            self.all_anchor_heights.append(heights)
            self.all_anchor_positions.append(anchor_positions)

        # self.image_size = image_size
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
            device=device,
        )

    def _get_anchor_positions(self, anchor_widths: torch.Tensor, anchor_heights: torch.Tensor, s: int, image_dim: int) -> torch.Tensor:
        """
        Get anchor positions for the volume in the shape:
        (1, feature_map_height*feature_map_width*anchor_heights*anchor_widths, 4)
        """
        x_step = image_dim / s
        y_step = image_dim / s

        grid = torch.zeros(s, s, len(anchor_heights), 4, device=self.device)

        x_grid_cell_centers = ((torch.arange(0, s, device=self.device).float() * x_step) + (x_step / 2)).reshape(1, s, 1)
        y_grid_cell_centers = ((torch.arange(0, s, device=self.device).float() * y_step) + (y_step / 2)).reshape(s, 1, 1)

        anchor_widths_grid = anchor_widths.reshape(1, 1, len(anchor_widths))
        anchor_heights_grid = anchor_heights.reshape(1, 1, len(anchor_heights))

        # x and y centers broadcast from ( s, 1, 1) to (s, s, 9)
        # widths and heights broadcast from ( 1, 1, 9) to ( s, s, 9)
        grid[:, :, :, 0] = x_grid_cell_centers - anchor_widths_grid / 2  # (1, s, s, 9)
        grid[:, :, :, 1] = y_grid_cell_centers - anchor_heights_grid / 2
        grid[:, :, :, 2] = x_grid_cell_centers + anchor_widths_grid / 2
        grid[:, :, :, 3] = y_grid_cell_centers + anchor_heights_grid / 2

        reshaped_grid = grid.reshape(s * s * len(anchor_heights), 4)

        return reshaped_grid  # (s*s*9, 4)

    def train_step(self, epoch: int) -> None:
        self.model.train()

        # explicit is better than implicit
        rpn_objectness_loss = 0.0
        rpn_bbox_loss = 0.0
        rpn_total_loss = 0.0
        fast_rcnn_cls_loss = 0.0
        fast_rcnn_bbox_loss = 0.0
        fast_rcnn_total_loss = 0.0
        faster_rcnn_total_loss = 0.0

        # num_correct = 0
        # num_incorrect_localization = 0
        # num_incorrect_other = 0
        # num_incorrect_background = 0
        # num_predictions = 0
        # num_objects = 0  # number of objects in the batch

        for batch, (image, raw_cls_gt, raw_bbox_gt, num_gt_bbox_in_each_image, metadata) in tqdm(
            enumerate(self.train_dataloader), total=len(self.train_dataloader), desc="Train Step", leave=False
        ):
            image, raw_cls_gt, raw_bbox_gt = image.to(self.device), raw_cls_gt.to(self.device), raw_bbox_gt.to(self.device)
            image, raw_cls_gt, raw_bbox_gt = cast(torch.Tensor, image), cast(torch.Tensor, raw_cls_gt), cast(torch.Tensor, raw_bbox_gt)

            total_faster_rcnn_loss = torch.tensor(0.0, device=self.device)

            fpn_maps = self.backbone(image)

            for fpn_map, anchor_heights, anchor_widths, anchor_positions in zip(
                fpn_maps, self.all_anchor_heights, self.all_anchor_widths, self.all_anchor_positions
            ):
                (
                    rpn_objectness_pred,
                    rpn_bbox_offset_pred,
                    fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox,
                    fast_rcnn_bbox_pred_for_some_rpn_bbox,
                    rpn_objectness_gt,
                    rpn_bbox_offset_gt,
                    fast_rcnn_cls_gt_nms_fg_and_bg_some,
                    fast_rcnn_bbox_gt_nms_fg_and_bg_some,
                ) = self.model(fpn_map, anchor_heights, anchor_widths, anchor_positions, raw_cls_gt, raw_bbox_gt)

                # fast_rcnn_cls_pred: tuple[]
                loss_dict = self.loss_fn(
                    rpn_objectness_pred,
                    rpn_bbox_offset_pred,
                    fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox,
                    fast_rcnn_bbox_pred_for_some_rpn_bbox,
                    rpn_objectness_gt,
                    rpn_bbox_offset_gt,
                    fast_rcnn_cls_gt_nms_fg_and_bg_some,
                    fast_rcnn_bbox_gt_nms_fg_and_bg_some,
                    device=self.device,
                )

                rpn_objectness_loss += loss_dict["rpn_objectness_loss"].item()
                rpn_bbox_loss += loss_dict["rpn_bbox_loss"].item()
                rpn_total_loss += loss_dict["rpn_total_loss"].item()
                fast_rcnn_cls_loss += loss_dict["fast_rcnn_cls_loss"].item()
                fast_rcnn_bbox_loss += loss_dict["fast_rcnn_bbox_loss"].item()
                fast_rcnn_total_loss += loss_dict["fast_rcnn_total_loss"].item()
                faster_rcnn_total_loss += loss_dict["faster_rcnn_total_loss"].item()

                total_faster_rcnn_loss += loss_dict["faster_rcnn_total_loss"]

            self.optimizer.zero_grad()
            total_faster_rcnn_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # result_dict = self.metric.compute_values(fast_rcnn_cls_pred, fast_rcnn_bbox_pred, gt_cls, gt_bbox)

            # num_correct += result_dict["num_correct"]
            # num_incorrect_localization += result_dict["num_incorrect_localization"]
            # num_incorrect_other += result_dict["num_incorrect_other"]
            # num_incorrect_background += result_dict["num_incorrect_background"]
            # num_objects += result_dict["num_objects"]

            # log_gradients(self.model)

            # doing a safe division here

            if batch != 0 and batch % self.log_interval == 0:
                self.run_manager.log_metrics(
                    {
                        "train/rpn_objectness_loss": rpn_objectness_loss,
                        "train/rpn_bbox_loss": rpn_bbox_loss,
                        "train/rpn_total_loss": rpn_total_loss,
                        "train/fast_rcnn_cls_loss": fast_rcnn_cls_loss,
                        "train/fast_rcnn_bbox_loss": fast_rcnn_bbox_loss,
                        "train/fast_rcnn_total_loss": fast_rcnn_total_loss,
                        "train/faster_rcnn_total_loss": faster_rcnn_total_loss,
                        # "train/accuracy": num_correct / num_objects,
                        # "train/percent_incorrect_localization": num_incorrect_localization / num_objects,
                        # "train/percent_incorrect_other": num_incorrect_other / num_objects,
                        # "train/percent_incorrect_background": num_incorrect_background / num_objects,
                    },
                    epoch + batch / len(self.train_dataloader),
                )

                rpn_objectness_loss = 0.0
                rpn_bbox_loss = 0.0
                rpn_total_loss = 0.0
                fast_rcnn_cls_loss = 0.0
                fast_rcnn_bbox_loss = 0.0
                fast_rcnn_total_loss = 0.0
                faster_rcnn_total_loss = 0.0

                # num_correct = 0
                # num_incorrect_localization = 0
                # num_incorrect_other = 0
                # num_incorrect_background = 0
                # num_predictions = 0
                # num_objects = 0

    def test_step(self, epoch: int) -> None:
        self.model.eval()

        rpn_objectness_loss = 0.0
        rpn_bbox_loss = 0.0
        rpn_total_loss = 0.0
        fast_rcnn_cls_loss = 0.0
        fast_rcnn_bbox_loss = 0.0
        fast_rcnn_total_loss = 0.0
        faster_rcnn_total_loss = 0.0

        num_correct = 0
        num_incorrect_localization = 0
        num_incorrect_other = 0
        num_incorrect_background = 0

        with torch.inference_mode():
            for batch, (image, raw_cls_gt, raw_bbox_gt, num_gt_bbox_in_each_image, metadata) in tqdm(
                enumerate(self.train_dataloader), total=len(self.train_dataloader), desc="Train Step", leave=False
            ):
                image, raw_cls_gt, raw_bbox_gt = image.to(self.device), raw_cls_gt.to(self.device), raw_bbox_gt.to(self.device)
                image, raw_cls_gt, raw_bbox_gt = cast(torch.Tensor, image), cast(torch.Tensor, raw_cls_gt), cast(torch.Tensor, raw_bbox_gt)

                fpn_maps = self.backbone(image)

                for fpn_map, anchor_heights, anchor_widths, anchor_positions in zip(
                    fpn_maps, self.all_anchor_heights, self.all_anchor_widths, self.all_anchor_positions
                ):
                    (
                        rpn_objectness_pred,
                        rpn_bbox_offset_pred,
                        fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox,
                        fast_rcnn_bbox_pred_for_some_rpn_bbox,
                        rpn_objectness_gt,
                        rpn_bbox_offset_gt,
                        fast_rcnn_cls_gt_nms_fg_and_bg_some,
                        fast_rcnn_bbox_gt_nms_fg_and_bg_some,
                    ) = self.model(fpn_map, anchor_heights, anchor_widths, anchor_positions, raw_cls_gt, raw_bbox_gt)

                    # fast_rcnn_cls_pred: tuple[]
                    loss_dict = self.loss_fn(
                        rpn_objectness_pred,
                        rpn_bbox_offset_pred,
                        fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox,
                        fast_rcnn_bbox_pred_for_some_rpn_bbox,
                        rpn_objectness_gt,
                        rpn_bbox_offset_gt,
                        fast_rcnn_cls_gt_nms_fg_and_bg_some,
                        fast_rcnn_bbox_gt_nms_fg_and_bg_some,
                        device=self.device,
                    )

                    rpn_objectness_loss += loss_dict["rpn_objectness_loss"].item()
                    rpn_bbox_loss += loss_dict["rpn_bbox_loss"].item()
                    rpn_total_loss += loss_dict["rpn_total_loss"].item()
                    fast_rcnn_cls_loss += loss_dict["fast_rcnn_cls_loss"].item()
                    fast_rcnn_bbox_loss += loss_dict["fast_rcnn_bbox_loss"].item()
                    fast_rcnn_total_loss += loss_dict["fast_rcnn_total_loss"].item()
                    faster_rcnn_total_loss += loss_dict["faster_rcnn_total_loss"].item()

                # num_correct += result_dict["num_correct"]
                # num_incorrect_localization += result_dict["num_incorrect_localization"]
                # num_incorrect_other += result_dict["num_incorrect_other"]
                # num_incorrect_background += result_dict["num_incorrect_background"]
                # num_objects += result_dict["num_objects"]

        self.run_manager.log_metrics(
            {
                "val/rpn_objectness_loss": rpn_objectness_loss,
                "val/rpn_bbox_loss": rpn_bbox_loss,
                "val/rpn_total_loss": rpn_total_loss,
                "val/fast_rcnn_cls_loss": fast_rcnn_cls_loss,
                "val/fast_rcnn_bbox_loss": fast_rcnn_bbox_loss,
                "val/fast_rcnn_total_loss": fast_rcnn_total_loss,
                "val/faster_rcnn_total_loss": faster_rcnn_total_loss,
                "val/accuracy": num_correct,
                "val/percent_incorrect_localization": num_incorrect_localization,
                "val/percent_incorrect_other": num_incorrect_other,
                "val/percent_incorrect_background": num_incorrect_background,
            },
            epoch,
        )

    def train(self) -> None:
        """
        Train a PyTorch model.

        Args:
            model: A PyTorch model to train.
            train_dataloader: A PyTorch DataLoader for training data.
            val_dataloader: A PyTorch DataLoader for validation data.
            lr_scheduler: A PyTorch learning rate scheduler.
            optimizer: A PyTorch optimizer to use for training.
            loss_fn: A PyTorch loss function to use for training.
            epoch_start: The starting epoch for training.
            epoch_end: The ending epoch for training.
            run_manager: An instance of the RunManager class for logging metrics.
            checkpoint_interval: The interval at which to save model checkpoints.
            log_interval: The interval at which to log metrics.
            device: The device to run the model on.
        """

        self.run_manager.log_metrics({"learning_rate": self.optimizer.param_groups[0]["lr"]}, self.epoch_start)

        for epoch in tqdm(range(self.epoch_start, self.epoch_end), desc="Epochs"):
            self.train_step(epoch)

            self.run_manager.log_metrics({"learning_rate": self.optimizer.param_groups[0]["lr"]}, epoch + 1)

            # saves model/epoch_5 at the end of epoch 5. epochs are 0 indexed.
            if epoch % self.checkpoint_interval == 0 or epoch == self.epoch_end - 1:
                self.test_step(epoch + 1)
                self.run_manager.save_model(self.model, epoch)

            self.lr_scheduler.step()
