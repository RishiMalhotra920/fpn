import os
import sys
from pathlib import Path
from typing import Optional

import torch
import typer
import yaml
from torchvision.transforms import v2 as transforms_v2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402

from fpn.checkpoint_loader import load_checkpoint
from fpn.data import BACKGROUND_CLASS_INDEX, CustomVOCDetectionDataset
from fpn.loss import FasterRCNNLoss
from fpn.lr_scheduler import get_custom_lr_scheduler, get_fixed_lr_scheduler
from fpn.models import FasterRCNNWithFPN
from fpn.run_manager import RunManager
from fpn.YOLO_metrics import YOLOMetrics

# from typing_extensions

config = yaml.safe_load(open("config.yaml"))

# to call this script, run the following command:
# start with learning rate 0.01 to speed the fuck out of the training. if it starts to bounce around, then we can decrease it.
# python train.py --num_epochs 10 --batch_size 32 --hidden_units 128 --learning_rate 0.01 --run_name cpu_run_on_image_net

# GPU training command:
# python train.py --num_epochs 50 --batch_size 128 --hidden_units 256 --learning_rate 0.001 --run_name cuda_run_with_256_hidden_units --device cuda

# python train.py --num_epochs 100 --batch_size 1024 --hidden_units 256 --learning_rate 0.005 --run_name image_net_train_deeper_network_and_dropout --device cuda

app = typer.Typer(
    name="Object Detection with FPN and Faster RCNN",
    help="Train an object detection model",
    epilog="Enjoy the program :)",
    pretty_exceptions_show_locals=False,
)


# python train.py --num-epochs 2 --batch-size 2 --lr-scheduler-name fixed --lr 0.001 --dropout 0.9 --run-name new-run --checkpoint-interval 5


@app.command()
def main(
    num_epochs: int = typer.Option(..., help="Number of epochs to train the model"),
    batch_size: int = typer.Option(..., help="Batch size for training the model"),
    lr_scheduler_name: str = typer.Option(..., help="Scheduler for the optimizer or custom"),
    lr: float = typer.Option(..., help="Learning rate for fixed scheduler or starting learning rate for custom scheduler"),
    nms_threshold: float = typer.Option(0.5, help="Non-maximum suppression threshold"),
    dropout: float = typer.Option(..., help="Dropout rate for the model"),
    run_name: str = typer.Option(None, help="A name for the run"),
    checkpoint_interval: int = typer.Option(1, help="The number of epochs to wait before saving model checkpoint"),
    image_dim: int = typer.Option(224, help="Size of the image"),
    continue_from_checkpoint_signature: Optional[str] = typer.Option(
        None, help="Checkpoint signature to continue training from eg: RunId:CheckpointPath eg: IM-23:checkpoints/epoch_10.pth"
    ),
    log_interval: int = typer.Option(10, help="The number of batches to wait before logging training status"),
    device: str = typer.Option("cpu", help="Device to train the model on"),
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
        faster_rcnn_with_fpn_model = FasterRCNNWithFPN((image_dim, image_dim), nms_threshold)

        run_manager = RunManager()  # empty run for testing!

        # run_manager = RunManager(
        #     new_run_name=run_name,
        #     source_files=[
        #         "../fpn/lr_scheduler.py",
        #         "../fpn/models/*.py",
        #         "../fpn/trainer.py",
        #         "train.py",
        #         "../fpn/loss/*.py",
        #     ],
        # )

        # switch to DistributedDataParallel if you have the heart for it!
        model = torch.nn.DataParallel(faster_rcnn_with_fpn_model)

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

        loss_fn = FasterRCNNLoss(BACKGROUND_CLASS_INDEX)

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
            "dropout": dropout,
            "num_workers": num_workers,
            "command": " ".join(sys.argv),
        }

        run_manager.log_data(
            {
                "parameters": parameters,
                "model/summary": str(model),
            }
        )

        this_metric = YOLOMetrics()

        # trainer = Trainer(
        #     model=model,
        #     train_dataloader=train_dataloader,
        #     val_dataloader=test_dataloader,
        #     lr_scheduler=lr_scheduler,
        #     optimizer=optimizer,
        #     loss_fn=loss_fn,
        #     metric=this_metric,
        #     epoch_start=epoch_start,
        #     epoch_end=epoch_start + num_epochs,
        #     run_manager=run_manager,
        #     checkpoint_interval=checkpoint_interval,
        #     log_interval=log_interval,
        #     device=device,
        # )

        # trainer.train()

        run_manager.end_run()
    except KeyboardInterrupt:
        typer.echo("Training interrupted by user")
    assert device in ["cpu", "cuda"], "Invalid device"


if __name__ == "__main__":
    app()
