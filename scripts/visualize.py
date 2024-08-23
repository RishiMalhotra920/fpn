import os
import sys
from typing import Optional

import torch
import typer
import yaml
from torchvision.transforms import v2 as transforms_v2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402


from fpn.data import CustomVOCDetectionDataset
from fpn.models import FPN, FasterRCNN
from fpn.utils import visualize_bbox
from fpn.utils.checkpoint_loader import load_checkpoint
from fpn.utils.predict_with_faster_rcnn import predict_with_faster_rcnn

config = yaml.safe_load(open("config.yaml"))

app = typer.Typer(
    name="Object Detection with FPN and Faster RCNN",
    help="Train an object detection model",
    epilog="Enjoy the program :)",
    pretty_exceptions_show_locals=False,
)


@app.command()
def main(
    backbone_checkpoint_signature: Optional[str] = typer.Option(
        None, help="Checkpoint signature to continue training from eg: RunId:CheckpointPath eg: FPNWIT-23:checkpoints/fpn_backbone/epoch_10.pth"
    ),
    faster_rcnn_model_checkpoint_signature: str = typer.Option(
        None, help="Checkpoint signature to continue training from eg: RunId:CheckpointPath eg: FPNWIT-23:checkpoints/faster_rcnn/epoch_10.pth"
    ),
    num_images: int = typer.Option(4, help="Number of images to predict on"),
    nms_threshold: float = typer.Option(0.5, help="Non-maximum suppression threshold for fast rcnn classifier final outputs"),
    image_dim: int = typer.Option(800, help="Size of the image"),
    num_rpn_rois_to_sample: int = typer.Option(256, help="Total number of RPN RoIs to sample across all feature map scales!"),
    rpn_pos_to_neg_ratio: float = typer.Option(1.0, help="Number of positive RoIs to sample for RPN"),
    rpn_pos_iou: float = typer.Option(0.7, help="When sampling rpn preds, we sample fg samples from matches with iou threshold > rpn_pos_iou"),
    rpn_neg_iou: float = typer.Option(0.3, help="When sampling rpn preds, we sample bg samples from matches with iou threshold < rpn_neg_iou"),
    device: str = typer.Option("cpu", help="Device to train the model on"),
):
    """
    --lr_scheduler
    """

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

    train_dataloader = train_dataset.get_dataloader(num_images, num_workers, shuffle=True)
    test_dataloader = val_dataset.get_dataloader(num_images, num_workers, shuffle=False)

    # Create model with help from model_builder.py
    # make it so that we minimize the sum of all the losses in the fpn!!
    backbone = FPN(False, device=device).to(device)
    faster_rcnn_model = FasterRCNN(
        (image_dim, image_dim),
        nms_threshold=nms_threshold,
        # divide by three because we run this faster rcnn for three feature maps
        fast_rcnn_dropout=0.0,
        num_rpn_rois_to_sample=num_rpn_rois_to_sample // 3,
        rpn_pos_to_neg_ratio=rpn_pos_to_neg_ratio,
        rpn_pos_iou=rpn_pos_iou,
        rpn_neg_iou=rpn_neg_iou,
        device=device,
    )

    if backbone_checkpoint_signature is not None:
        load_checkpoint(backbone, backbone_checkpoint_signature)

    if faster_rcnn_model_checkpoint_signature is not None:
        load_checkpoint(faster_rcnn_model, faster_rcnn_model_checkpoint_signature)

    backbone.eval()
    faster_rcnn_model.eval()

    images, raw_cls_gt, raw_bbox_gt, num_gt_bbox_in_each_image, metadata = next(iter(train_dataloader))
    rpn_picked_label_pred, rpn_picked_bbox_pred = predict_with_faster_rcnn(
        images, image_dim, backbone, faster_rcnn_model, raw_cls_gt, raw_bbox_gt, device=device
    )

    visualize_bbox(images, rpn_picked_label_pred, rpn_picked_bbox_pred, raw_cls_gt, raw_bbox_gt, show_pred=True, show_gt=True)

    print("wait here")


if __name__ == "__main__":
    app()
