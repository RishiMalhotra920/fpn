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
from fpn.utils.anchor_utils import create_anchors


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
        epoch_start: int,
        epoch_end: int,
        run_manager: RunManager,
        checkpoint_interval: int,
        log_interval: int,
        *,
        device: str,
        image_size: tuple[int, int],
        feature_map_dims: list[int],
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
        self.backbone = backbone

        self.all_anchor_widths, self.all_anchor_heights, self.all_anchor_positions = create_anchors(image_size, feature_map_dims, device=device)
        self.rpn_recall_iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

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
        # the first time we log, we log for 11 batches so need num_batches for averaging
        num_batches = 0
        rpn_recall_iou_metrics = {f"rpn_recall@{t}": 0.0 for t in self.rpn_recall_iou_thresholds}
        total_rpn_num_fg_bbox_picked = 0.0
        total_rpn_num_bg_bbox_picked = 0.0

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

            # list[num_fpn_maps, list[num_images, tensor(L_i, 4)]]
            # all_rpn_bbox_pred_nms_fg_and_bg_some = []

            for fpn_map, anchor_heights, anchor_widths, anchor_positions in zip(
                fpn_maps, self.all_anchor_heights, self.all_anchor_widths, self.all_anchor_positions
            ):
                (
                    rpn_objectness_pred,
                    rpn_bbox_offset_pred,
                    rpn_bbox_pred_nms_fg_and_bg_some,
                    rpn_bbox_pred_nms_fg_and_bg_some_fg_mask,
                    # fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox,
                    # fast_rcnn_bbox_offsets_pred,
                    rpn_objectness_gt,
                    rpn_bbox_offset_gt,
                    # fast_rcnn_cls_gt_nms_fg_and_bg_some,
                    # fast_rcnn_bbox_offsets_gt,
                    rpn_num_fg_bbox_picked,
                    rpn_num_bg_bbox_picked,
                ) = self.model(fpn_map, anchor_heights, anchor_widths, anchor_positions, raw_cls_gt, raw_bbox_gt)

                # fast_rcnn_cls_pred: tuple[]
                loss_dict = self.loss_fn(
                    rpn_objectness_pred,
                    rpn_bbox_offset_pred,
                    # fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox,
                    # fast_rcnn_bbox_offsets_pred,
                    rpn_objectness_gt,
                    rpn_bbox_offset_gt,
                    # fast_rcnn_cls_gt_nms_fg_and_bg_some,
                    # fast_rcnn_bbox_offsets_gt,
                    device=self.device,
                )

                # concatenate
                # all_rpn_bbox_pred_nms_fg_and_bg_some.append(rpn_bbox_pred_nms_fg_and_bg_some)

                # [torch.cat((a, b), dim=0) for a, b in zip(all_rpn_bbox_pred_nms_fg_and_bg_some, rpn_bbox_offset_gt)]

                rpn_objectness_loss += loss_dict["rpn_objectness_loss"].item()
                rpn_bbox_loss += loss_dict["rpn_bbox_loss"].item()
                rpn_total_loss += loss_dict["rpn_total_loss"].item()
                # fast_rcnn_cls_loss += loss_dict["fast_rcnn_cls_loss"].item()
                # fast_rcnn_bbox_loss += loss_dict["fast_rcnn_bbox_loss"].item()
                # fast_rcnn_total_loss += loss_dict["fast_rcnn_total_loss"].item()
                # faster_rcnn_total_loss += loss_dict["faster_rcnn_total_loss"].item()
                # divide by 3 because we have 3 fpn maps

                total_faster_rcnn_loss += loss_dict["faster_rcnn_total_loss"]

                total_rpn_num_bg_bbox_picked += rpn_num_bg_bbox_picked
                total_rpn_num_fg_bbox_picked += rpn_num_fg_bbox_picked

            # # list[num_images, tensor(L_i_fpn_map_1+L_i_fpn_map2+L_i_fpn_map3, 4)]
            # all_rpn_bbox_pred_nms_fg_and_bg_some_per_image = [
            #     torch.cat((t1, t2, t3))
            #     for t1, t2, t3 in zip(
            #         all_rpn_bbox_pred_nms_fg_and_bg_some[0],
            #         all_rpn_bbox_pred_nms_fg_and_bg_some[1],
            #         all_rpn_bbox_pred_nms_fg_and_bg_some[2],
            #     )
            # ]

            # rpn_metrics = calculate_rpn_metrics(all_rpn_bbox_pred_nms_fg_and_bg_some_per_image, raw_bbox_gt, self.rpn_recall_iou_thresholds)
            # rpn_recall_iou_metrics = {k: v + rpn_metrics[k] for k, v in rpn_recall_iou_metrics.items()}

            num_batches += 1

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
                # print("epoch", epoch, "batch", batch, "fast_rcnn_cls_loss", fast_rcnn_cls_loss)
                rpn_recall_iou_metrics = {k: v / num_batches for k, v in rpn_recall_iou_metrics.items()}
                self.run_manager.log_metrics(
                    {
                        "train/rpn_objectness_loss": rpn_objectness_loss / num_batches,
                        "train/rpn_bbox_loss": rpn_bbox_loss / num_batches,
                        "train/rpn_total_loss": rpn_total_loss / num_batches,
                        # "train/fast_rcnn_cls_loss": fast_rcnn_cls_loss / num_batches,
                        # "train/fast_rcnn_bbox_loss": fast_rcnn_bbox_loss / num_batches,
                        # "train/fast_rcnn_total_loss": fast_rcnn_total_loss / num_batches,
                        # "train/faster_rcnn_total_loss": faster_rcnn_total_loss / num_batches,
                        "train/rpn_num_fg_bbox_picked": total_rpn_num_fg_bbox_picked / num_batches,
                        "train/rpn_num_bg_bbox_picked": total_rpn_num_bg_bbox_picked / num_batches,
                        # "train/accuracy": num_correct / num_objects, #TODO: remember to divide by num_batches
                        # "train/percent_incorrect_localization": num_incorrect_localization / num_objects,
                        # "train/percent_incorrect_other": num_incorrect_other / num_objects,
                        # "train/percent_incorrect_background": num_incorrect_background / num_objects,
                    },
                    # | {f"train_{k}": v for k, v in rpn_recall_iou_metrics.items()},
                    epoch + batch / len(self.train_dataloader),
                )

                rpn_objectness_loss = 0.0
                rpn_bbox_loss = 0.0
                rpn_total_loss = 0.0
                fast_rcnn_cls_loss = 0.0
                fast_rcnn_bbox_loss = 0.0
                fast_rcnn_total_loss = 0.0
                faster_rcnn_total_loss = 0.0
                rpn_recall_iou_metrics = {f"rpn_recall@{t}": 0.0 for t in self.rpn_recall_iou_thresholds}
                total_rpn_num_fg_bbox_picked = 0
                total_rpn_num_bg_bbox_picked = 0
                num_batches = 0

                # num_correct = 0
                # num_incorrect_localization = 0
                # num_incorrect_other = 0
                # num_incorrect_background = 0
                # num_predictions = 0
                # num_objects = 0

    # def test_step(self, epoch: int) -> None:
    #     self.model.eval()

    #     rpn_objectness_loss = 0.0
    #     rpn_bbox_loss = 0.0
    #     rpn_total_loss = 0.0
    #     fast_rcnn_cls_loss = 0.0
    #     fast_rcnn_bbox_loss = 0.0
    #     fast_rcnn_total_loss = 0.0
    #     faster_rcnn_total_loss = 0.0

    #     num_correct = 0
    #     num_incorrect_localization = 0
    #     num_incorrect_other = 0
    #     num_incorrect_background = 0

    #     with torch.inference_mode():
    #         for batch, (image, raw_cls_gt, raw_bbox_gt, num_gt_bbox_in_each_image, metadata) in tqdm(
    #             enumerate(self.train_dataloader), total=len(self.train_dataloader), desc="Train Step", leave=False
    #         ):
    #             image, raw_cls_gt, raw_bbox_gt = image.to(self.device), raw_cls_gt.to(self.device), raw_bbox_gt.to(self.device)
    #             image, raw_cls_gt, raw_bbox_gt = cast(torch.Tensor, image), cast(torch.Tensor, raw_cls_gt), cast(torch.Tensor, raw_bbox_gt)

    #             fpn_maps = self.backbone(image)

    #             for fpn_map, anchor_heights, anchor_widths, anchor_positions in zip(
    #                 fpn_maps, self.all_anchor_heights, self.all_anchor_widths, self.all_anchor_positions
    #             ):
    #                 (
    #                     rpn_objectness_pred,
    #                     rpn_bbox_offset_pred,
    #                     fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox,
    #                     fast_rcnn_bbox_pred_for_some_rpn_bbox,
    #                     rpn_objectness_gt,
    #                     rpn_bbox_offset_gt,
    #                     fast_rcnn_cls_gt_nms_fg_and_bg_some,
    #                     fast_rcnn_bbox_gt_nms_fg_and_bg_some,
    #                 ) = self.model(fpn_map, anchor_heights, anchor_widths, anchor_positions, raw_cls_gt, raw_bbox_gt)

    #                 # fast_rcnn_cls_pred: tuple[]
    #                 loss_dict = self.loss_fn(
    #                     rpn_objectness_pred,
    #                     rpn_bbox_offset_pred,
    #                     fast_rcnn_cls_probs_for_all_classes_for_some_rpn_bbox,
    #                     fast_rcnn_bbox_pred_for_some_rpn_bbox,
    #                     rpn_objectness_gt,
    #                     rpn_bbox_offset_gt,
    #                     fast_rcnn_cls_gt_nms_fg_and_bg_some,
    #                     fast_rcnn_bbox_gt_nms_fg_and_bg_some,
    #                     device=self.device,
    #                 )

    #                 rpn_objectness_loss += loss_dict["rpn_objectness_loss"].item()
    #                 rpn_bbox_loss += loss_dict["rpn_bbox_loss"].item()
    #                 rpn_total_loss += loss_dict["rpn_total_loss"].item()
    #                 fast_rcnn_cls_loss += loss_dict["fast_rcnn_cls_loss"].item()
    #                 fast_rcnn_bbox_loss += loss_dict["fast_rcnn_bbox_loss"].item()
    #                 fast_rcnn_total_loss += loss_dict["fast_rcnn_total_loss"].item()
    #                 faster_rcnn_total_loss += loss_dict["faster_rcnn_total_loss"].item()

    #             # num_correct += result_dict["num_correct"]
    #             # num_incorrect_localization += result_dict["num_incorrect_localization"]
    #             # num_incorrect_other += result_dict["num_incorrect_other"]
    #             # num_incorrect_background += result_dict["num_incorrect_background"]
    #             # num_objects += result_dict["num_objects"]

    #     self.run_manager.log_metrics(
    #         {
    #             "val/rpn_objectness_loss": rpn_objectness_loss,
    #             "val/rpn_bbox_loss": rpn_bbox_loss,
    #             "val/rpn_total_loss": rpn_total_loss,
    #             "val/fast_rcnn_cls_loss": fast_rcnn_cls_loss,
    #             "val/fast_rcnn_bbox_loss": fast_rcnn_bbox_loss,
    #             "val/fast_rcnn_total_loss": fast_rcnn_total_loss,
    #             "val/faster_rcnn_total_loss": faster_rcnn_total_loss,
    #             "val/accuracy": num_correct,
    #             "val/percent_incorrect_localization": num_incorrect_localization,
    #             "val/percent_incorrect_other": num_incorrect_other,
    #             "val/percent_incorrect_background": num_incorrect_background,
    #         },
    #         epoch,
    #     )

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
            # for testing
            # self.run_manager.save_model("checkpoints/faster_rcnn", self.model, epoch)
            # self.run_manager.save_model("checkpoints/fpn_backbone", self.backbone, epoch)

            self.train_step(epoch)

            self.run_manager.log_metrics({"learning_rate": self.optimizer.param_groups[0]["lr"]}, epoch + 1)

            # saves model/epoch_5 at the end of epoch 5. epochs are 0 indexed.
            if epoch % self.checkpoint_interval == 0 or epoch == self.epoch_end - 1:
                # pass
                # self.test_step(epoch + 1)
                self.run_manager.save_model("checkpoints/faster_rcnn", self.model, epoch)
                self.run_manager.save_model("checkpoints/fpn_backbone", self.backbone, epoch)

            self.lr_scheduler.step()
