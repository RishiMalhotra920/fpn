import os
import sys
from pathlib import Path
from typing import Optional

import torch
import typer
import yaml
from torchvision.transforms import v2 as transforms_v2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402

from fpn.data import BACKGROUND_CLASS_INDEX, CustomVOCDetectionDataset
from fpn.loss import FasterRCNNLoss
from fpn.lr_scheduler import get_custom_lr_scheduler, get_fixed_lr_scheduler
from fpn.models import FPN, FasterRCNN
from fpn.run_manager import RunManager
from fpn.trainer import Trainer
from fpn.utils.checkpoint_loader import load_checkpoint

config = yaml.safe_load(open("config.yaml"))


app = typer.Typer(
    name="Object Detection with FPN and Faster RCNN",
    help="Train an object detection model",
    epilog="Enjoy the program :)",
    pretty_exceptions_show_locals=False,
)


# python train.py --num-epochs 128 --batch-size 128 --lr-scheduler-name fixed --lr 0.0001 --fast-rcnn-dropout 0.0 --run-name double_weight_on_bbox_loss --checkpoint-interval 5 --no-freeze-backbone --device cuda


@app.command()
def main(
    num_epochs: int = typer.Option(..., help="Number of epochs to train the model"),
    batch_size: int = typer.Option(..., help="Batch size for training the model"),
    lr_scheduler_name: str = typer.Option(..., help="Scheduler for the optimizer or custom"),
    lr: float = typer.Option(..., help="Learning rate for fixed scheduler or starting learning rate for custom scheduler"),
    nms_threshold: float = typer.Option(0.5, help="Non-maximum suppression threshold for fast rcnn classifier final outputs"),
    fast_rcnn_dropout: float = typer.Option(..., help="Dropout rate for the fast rcnn classifier"),
    freeze_backbone: bool = typer.Option(True, help="Freeze the backbone"),
    run_name: str = typer.Option(None, help="A name for the run"),
    checkpoint_interval: int = typer.Option(1, help="The number of epochs to wait before saving model checkpoint"),
    image_dim: int = typer.Option(800, help="Size of the image"),
    continue_from_checkpoint_signature: Optional[str] = typer.Option(
        None, help="Checkpoint signature to continue training from eg: RunId:CheckpointPath eg: IM-23:checkpoints/epoch_10.pth"
    ),
    log_interval: int = typer.Option(10, help="The number of batches to wait before logging training status"),
    device: str = typer.Option("cpu", help="Device to train the model on"),
    num_rpn_rois_to_sample: int = typer.Option(256, help="Total number of RPN RoIs to sample across all feature map scales!"),
    rpn_pos_to_neg_ratio: int = typer.Option(1.0, help="Number of positive RoIs to sample for RPN"),
    rpn_pos_iou: float = typer.Option(0.5, help="When sampling rpn preds, we sample fg samples from matches with iou threshold > rpn_pos_iou"),
    rpn_neg_iou: float = typer.Option(0.3, help="When sampling rpn gts, we sample bg samples from matches with iou threshold < rpn_neg_iou"),
    lambda_rpn_objectness: int = typer.Option(1, help="Weight for rpn objectness loss"),
    lambda_rpn_bbox: int = typer.Option(20, help="Weight for rpn bbox loss"),
    lambda_fast_rcnn_cls: int = typer.Option(10, help="Weight for fast rcnn cls loss"),
    lambda_fast_rcnn_bbox: int = typer.Option(10, help="Weight for fast rcnn bbox loss"),
    train_dir: Path = typer.Option(Path(config["image_net_data_dir"]) / "train", help="Directory containing training data"),
    val_dir: Path = typer.Option(Path(config["image_net_data_dir"]) / "val", help="Directory containing validation data"),
):
    """
    --lr_scheduler
    """

    # Args validation
    if lr_scheduler_name not in ["custom", "fixed"]:
        print(lr_scheduler_name, "here")
        typer.echo("Error: Invalid lr_scheduler", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Starting training for run_name: {run_name}")

    # train
    try:
        if device == "cuda" and not torch.cuda.is_available():
            raise Exception("CUDA is not available on this device")

        # yolo uses 448x448 images
        # Create transforms

        data_transform = transforms_v2.Compose(
            [
                # transforms.RandomResizedCrop(50),
                transforms_v2.Resize((image_dim, image_dim)),  # YOLO used 448x448 images. here we use 224x224
                transforms_v2.RandomHorizontalFlip(),
                # translate x, y by up to 0.1ximage_width, 0.1ximage_height, scale by 1.0-1.2ximage_dim, rotate by -30 to 30 degrees
                transforms_v2.RandomAffine(degrees=(0, 30), translate=(0.1, 0.1), scale=(1.0, 1.2), shear=0),
                transforms_v2.ColorJitter(brightness=0.5, contrast=0.5),
                transforms_v2.ToImage(),  # convert to PIL image 0..255
                transforms_v2.ToDtype(torch.float, scale=True),  # convert to float32 and scale to 0..1
                transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.RandomErasing()
            ]
        )

        cpu_count = 9  # 9 worker threads only.
        num_workers = cpu_count if cpu_count is not None else 0

        train_dataset = CustomVOCDetectionDataset(config["pascal_voc_root_dir"], "train", data_transform)
        val_dataset = CustomVOCDetectionDataset(config["pascal_voc_root_dir"], "val", data_transform)

        train_dataloader = train_dataset.get_dataloader(batch_size, num_workers, shuffle=True)
        test_dataloader = val_dataset.get_dataloader(batch_size, num_workers, shuffle=False)

        # Create model with help from model_builder.py
        # make it so that we minimize the sum of all the losses in the fpn!!
        backbone = FPN(freeze_backbone, device=device).to(device)

        faster_rcnn_with_fpn_model = FasterRCNN(
            (image_dim, image_dim),
            nms_threshold=nms_threshold,
            # divide by three because we run this faster rcnn for three feature maps
            fast_rcnn_dropout=fast_rcnn_dropout,
            num_rpn_rois_to_sample=num_rpn_rois_to_sample // 3,
            rpn_pos_to_neg_ratio=rpn_pos_to_neg_ratio,
            rpn_pos_iou=rpn_pos_iou,
            rpn_neg_iou=rpn_neg_iou,
            device=device,
        )
        # faster_rcnn_with_fpn_model = FasterRCNN((image_dim, image_dim), nms_threshold, device=device)

        # run_manager = RunManager()  # empty run for testing!

        run_manager = RunManager(
            new_run_name=run_name,
            source_files=[
                "../fpn/lr_scheduler.py",
                "../fpn/models/*.py",
                "../fpn/trainer.py",
                "train.py",
                "../fpn/loss/*.py",
            ],
        )

        # switch to DistributedDataParallel if you have the heart for it!
        if device == "cuda":
            model = torch.nn.DataParallel(faster_rcnn_with_fpn_model)
        else:
            model = faster_rcnn_with_fpn_model  # type: ignore

        if continue_from_checkpoint_signature is not None:
            print("Loading YOLO checkpoint ...")
            epoch_start = load_checkpoint(model, checkpoint_signature=continue_from_checkpoint_signature)
            run_manager.add_tags(["run_continuation"])
            run_manager.set_checkpoint_to_continue_from(continue_from_checkpoint_signature)
            print("YOLO checkpoint loaded...")
        else:
            epoch_start = 0
            run_manager.add_tags(["new_run"])

        # Set loss and optimizer
        # loss_fn = torch.nn.CrossEntropyLoss()

        loss_fn = FasterRCNNLoss(
            BACKGROUND_CLASS_INDEX,
            lambda_rpn_objectness=lambda_rpn_objectness,
            lambda_rpn_bbox=lambda_rpn_bbox,
            lambda_fast_rcnn_cls=lambda_fast_rcnn_cls,
            lambda_fast_rcnn_bbox=lambda_fast_rcnn_bbox,
            device=device,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if lr_scheduler_name == "custom":
            lr_scheduler = get_custom_lr_scheduler(optimizer)
        else:
            lr_scheduler = get_fixed_lr_scheduler(optimizer)

        for epoch in range(epoch_start):
            # step the lr_scheduler to match with the current_epoch
            lr_scheduler.step()
            print("epoch", epoch, "lr", optimizer.param_groups[0]["lr"])

        print("this is lr_scheduler current lr", optimizer.param_groups[0]["lr"])

        parameters = {
            "num_epochs": num_epochs,
            "lr_scheduler": lr_scheduler,
            "starting_lr": lr,
            "batch_size": batch_size,
            "loss_fn": "YOLOLossv0",
            "optimizer": "Adam",
            "device": device,
            "fast_rcnn_dropout": fast_rcnn_dropout,
            "num_workers": num_workers,
            "command": " ".join(sys.argv),
        }

        run_manager.log_data({"parameters": parameters, "model/summary": str(model)})

        trainer = Trainer(
            backbone=backbone,
            model=model,  # type: ignore
            train_dataloader=train_dataloader,
            val_dataloader=test_dataloader,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch_start=epoch_start,
            epoch_end=epoch_start + num_epochs,
            run_manager=run_manager,
            checkpoint_interval=checkpoint_interval,
            log_interval=log_interval,
            image_size=(image_dim, image_dim),
            device=device,
        )

        trainer.train()

        run_manager.end_run()
    except KeyboardInterrupt:
        typer.echo("Training interrupted by user")
    assert device in ["cpu", "cuda"], "Invalid device"


if __name__ == "__main__":
    app()
