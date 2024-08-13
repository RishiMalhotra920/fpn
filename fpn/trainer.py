import torch
from torch import nn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from fpn.loss.faster_rcnn_loss import FasterRCNNLoss
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
        model: DataParallel,
        train_dataloader: DataLoader,
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
        device: str,
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

    def __repr__(self) -> str:
        return f"Trainer(model={self.model}, train_dataloader={self.train_dataloader}, val_dataloader={self.val_dataloader}, lr_scheduler={self.lr_scheduler}, optimizer={self.optimizer}, loss_fn={self.loss_fn}, metric={self.metric}, epoch_start={self.epoch_start}, epoch_end={self.epoch_end}, run_manager={self.run_manager}, checkpoint_interval={self.checkpoint_interval}, log_interval={self.log_interval}, device={self.device})"

    def train_step(self, epoch: int) -> None:
        self.model.train()

        # explicit is better than implicit
        rpn_objectness_loss = 0
        rpn_bbox_loss = 0
        rpn_total_loss = 0
        fast_rcnn_cls_loss = 0
        fast_rcnn_bbox_loss = 0
        fast_rcnn_total_loss = 0
        faster_rcnn_total_loss = 0

        num_correct = 0
        num_incorrect_localization = 0
        num_incorrect_other = 0
        num_incorrect_background = 0
        num_predictions = 0
        num_objects = 0  # number of objects in the batch

        for batch, (image, gt_cls, gt_bboxes, metadata) in tqdm(
            enumerate(self.train_dataloader), total=len(self.train_dataloader), desc="Train Step", leave=False
        ):
            image, gt_cls, gt_bboxes = image.to(self.device), gt_cls.to(self.device), gt_bboxes.to(self.device)
            # gt_cls: (B, gt_cls)
            # gt_bboxes: (B, gt_cls, 4)

            objectness_pred, rpn_bboxes_pred, foreground_objectness_pred, foreground_bboxes_pred, fast_rcnn_cls_pred, fast_rcnn_bboxes_pred = (
                self.model(image)
            )

            # objectness_pred: (B, nBB*1/pos_to_neg_ratio)
            # rpn_bboxes_pred: (B, nBB*1/pos_to_neg_ratio, 4)
            # foreground_objectness_pred: (B, nBB*pos_to_neg_ratio)
            # foreground_bboxes_pred: (B, k*pos_to_neg_ratio, 4)
            # fast_rcnn_cls_pred: (B, nBB)
            # fast_rcnn_bboxes_pred: (B, nBB, 4)

            loss_dict = self.loss_fn(
                objectness_pred,
                rpn_bboxes_pred,
                # RPN_BBOX_ANCHOR.expand_as(rpn_bboxes_pred),
                foreground_objectness_pred,
                foreground_bboxes_pred,
                fast_rcnn_cls_pred,
                fast_rcnn_bboxes_pred,
                gt_cls,
                gt_bboxes,
            )

            rpn_objectness_loss += loss_dict["rpn_objectness_loss"].item()
            rpn_bbox_loss += loss_dict["rpn_bbox_loss"].item()
            rpn_total_loss += loss_dict["rpn_total_loss"].item()
            fast_rcnn_cls_loss += loss_dict["fast_rcnn_cls_loss"].item()
            fast_rcnn_bbox_loss += loss_dict["fast_rcnn_bbox_loss"].item()
            fast_rcnn_total_loss += loss_dict["fast_rcnn_total_loss"].item()
            faster_rcnn_total_loss += loss_dict["faster_rcnn_total_loss"].item()

            self.optimizer.zero_grad()
            loss_dict["faster_rcnn_total_loss"].backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()

            result_dict = self.metric.compute_values(
                fast_rcnn_cls_pred.detach().cpu().numpy(),
                fast_rcnn_bboxes_pred.detach().cpu().numpy(),
                gt_cls.detach().cpu().numpy(),
                gt_bboxes.detach().cpu().numpy(),
            )

            num_correct += result_dict["num_correct"]
            num_incorrect_localization += result_dict["num_incorrect_localization"]
            num_incorrect_other += result_dict["num_incorrect_other"]
            num_incorrect_background += result_dict["num_incorrect_background"]
            num_objects += result_dict["num_objects"]

            num_predictions += len(fast_rcnn_cls_pred)

            log_gradients(self.model)

            if batch != 0 and batch % self.log_interval == 0:
                self.run_manager.log_metrics(
                    {
                        "train/rpn_objectness_loss": rpn_objectness_loss / num_predictions,
                        "train/rpn_bbox_loss": rpn_bbox_loss / num_predictions,
                        "train/rpn_total_loss": rpn_total_loss / num_predictions,
                        "train/fast_rcnn_cls_loss": fast_rcnn_cls_loss / num_predictions,
                        "train/fast_rcnn_bbox_loss": fast_rcnn_bbox_loss / num_predictions,
                        "train/fast_rcnn_total_loss": fast_rcnn_total_loss / num_predictions,
                        "train/faster_rcnn_total_loss": faster_rcnn_total_loss / num_predictions,
                        "train/accuracy": num_correct / num_objects,
                        "train/percent_incorrect_localization": num_incorrect_localization / num_objects,
                        "train/percent_incorrect_other": num_incorrect_other / num_objects,
                        "train/percent_incorrect_background": num_incorrect_background / num_objects,
                    },
                    epoch + batch / len(self.train_dataloader),
                )

                rpn_objectness_loss = 0
                rpn_bbox_loss = 0
                rpn_total_loss = 0
                fast_rcnn_cls_loss = 0
                fast_rcnn_bbox_loss = 0
                fast_rcnn_total_loss = 0
                faster_rcnn_total_loss = 0

                num_correct = 0
                num_incorrect_localization = 0
                num_incorrect_other = 0
                num_incorrect_background = 0
                num_predictions = 0
                num_objects = 0

    def test_step(self, epoch: int) -> None:
        self.model.eval()

        rpn_objectness_loss = 0
        rpn_bbox_loss = 0
        rpn_total_loss = 0
        fast_rcnn_cls_loss = 0
        fast_rcnn_bbox_loss = 0
        fast_rcnn_total_loss = 0
        faster_rcnn_total_loss = 0

        num_correct = 0
        num_incorrect_localization = 0
        num_incorrect_other = 0
        num_incorrect_background = 0
        num_predictions = 0
        num_objects = 0

        with torch.inference_mode():
            for batch, (image, gt_cls, gt_bboxes, metadata) in tqdm(
                enumerate(self.val_dataloader), total=len(self.val_dataloader), desc="Test Step", leave=False
            ):
                image, gt_cls, gt_bboxes = image.to(self.device), gt_cls.to(self.device), gt_bboxes.to(self.device)
                # gt_cls: (B, gt_cls)
                # gt_bboxes: (B, gt_cls, 4)

                objectness_pred, rpn_bboxes_pred, foreground_objectness_pred, foreground_bboxes_pred, fast_rcnn_cls_pred, fast_rcnn_bboxes_pred = (
                    self.model(image)
                )

                # objectness_pred: (B, nBB*1/pos_to_neg_ratio)
                # rpn_bboxes_pred: (B, nBB*1/pos_to_neg_ratio, 4)
                # foreground_objectness_pred: (B, nBB*pos_to_neg_ratio)
                # foreground_bboxes_pred: (B, k*pos_to_neg_ratio, 4)
                # fast_rcnn_cls_pred: (B, nBB)
                # fast_rcnn_bboxes_pred: (B, nBB, 4)

                loss_dict = self.loss_fn(
                    objectness_pred,
                    rpn_bboxes_pred,
                    foreground_objectness_pred,
                    foreground_bboxes_pred,
                    fast_rcnn_cls_pred,
                    fast_rcnn_bboxes_pred,
                    gt_cls,
                    gt_bboxes,
                )

                rpn_objectness_loss += loss_dict["rpn_objectness_loss"].item()
                rpn_bbox_loss += loss_dict["rpn_bbox_loss"].item()
                rpn_total_loss += loss_dict["rpn_total_loss"].item()
                fast_rcnn_cls_loss += loss_dict["fast_rcnn_cls_loss"].item()
                fast_rcnn_bbox_loss += loss_dict["fast_rcnn_bbox_loss"].item()
                fast_rcnn_total_loss += loss_dict["fast_rcnn_total_loss"].item()
                faster_rcnn_total_loss += loss_dict["faster_rcnn_total_loss"].item()

                result_dict = self.metric.compute_values(
                    fast_rcnn_cls_pred.detach().cpu().numpy(),
                    fast_rcnn_bboxes_pred.detach().cpu().numpy(),
                    gt_cls.detach().cpu().numpy(),
                    gt_bboxes.detach().cpu().numpy(),
                )

                num_correct += result_dict["num_correct"]
                num_incorrect_localization += result_dict["num_incorrect_localization"]
                num_incorrect_other += result_dict["num_incorrect_other"]
                num_incorrect_background += result_dict["num_incorrect_background"]
                num_objects += result_dict["num_objects"]

        # Note: if you average out the loss in the loss function, then you should divide by len(dataloader) here.
        self.run_manager.log_metrics(
            {
                "val/rpn_objectness_loss": rpn_objectness_loss / num_predictions,
                "val/rpn_bbox_loss": rpn_bbox_loss / num_predictions,
                "val/rpn_total_loss": rpn_total_loss / num_predictions,
                "val/fast_rcnn_cls_loss": fast_rcnn_cls_loss / num_predictions,
                "val/fast_rcnn_bbox_loss": fast_rcnn_bbox_loss / num_predictions,
                "val/fast_rcnn_total_loss": fast_rcnn_total_loss / num_predictions,
                "val/faster_rcnn_total_loss": faster_rcnn_total_loss / num_predictions,
                "val/accuracy": num_correct / num_objects,
                "val/percent_incorrect_localization": num_incorrect_localization / num_objects,
                "val/percent_incorrect_other": num_incorrect_other / num_objects,
                "val/percent_incorrect_background": num_incorrect_background / num_objects,
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
