import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402

import color_code_error_messages  # noqa: F401
from torchvision.transforms import v2 as transforms_v2

from src.checkpoint_loader import load_checkpoint
from src.data_setup.yolo_train_data_setup import create_datasets
from src.models import yolo_net
from src.utils import predict_on_random_pascal_voc_images

config = yaml.safe_load(open("config.yaml"))


def visualize(args):
    data_transform = transforms_v2.Compose(
        [
            transforms_v2.Resize((448, 448)),
            transforms_v2.RandomHorizontalFlip(),
            transforms_v2.RandomAffine(
                degrees=(0, 30), translate=(0.1, 0.1), scale=(1.0, 1.2), shear=0
            ),
            # transforms_v2.ColorJitter(brightness=0.5, contrast=0.5),
            transforms_v2.ToTensor(),
            # transforms_v2.Normalize(
            # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # ),
        ]
    )

    train_dataset, mini_val_dataset = create_datasets(
        config["pascal_voc_root_dir"], data_transform
    )

    yolo_net_model = yolo_net.YOLONet(dropout=0).to("cpu")
    model = torch.nn.DataParallel(yolo_net_model)

    if args.checkpoint_signature is not None:
        load_checkpoint(model, args.checkpoint_signature)

    predict_on_random_pascal_voc_images(
        model,
        mini_val_dataset,
        threshold=args.threshold,
        show_preds=args.show_preds,
        show_labels=args.show_labels,
        n=5,
        seed=args.seed,
        apply_nms=args.apply_nms,
        nms_iou_threshold=args.nms_iou_threshold,
    )

    # display_random_images(train_dataset,
    #   class_names=class_names, n=5, seed=4)


if __name__ == "__main__":
    # for example
    # python visualize_pascal_voc.py --checkpoint_signature IM-254:checkpoints/epoch_335 --threshold 0.5 --seed 420 --show_preds true
    # --hidden_units 256
    parser = argparse.ArgumentParser(description="Visualize the model's predictions")
    # parser.add_argument("--hidden_units", type=int,
    # help="The number of hidden units", required=True)
    parser.add_argument(
        "--checkpoint_signature",
        type=str,
        help="The path to the checkpoint in the format RUN_ID:CHECKPOINT_PATH eg: IM-28:checkpoints/epoch_5",
        required=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="The seed for the random number generator",
        required=False,
        default=42,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="The threshold for the bounding box",
        required=True,
    )
    parser.add_argument(
        "--show_labels",
        type=bool,
        help="Whether to show the labels on the bounding boxes",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--show_preds",
        type=bool,
        help="Whether to show the scores on the bounding boxes",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--apply_nms",
        type=bool,
        help="Whether to apply non-maximum suppression",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--nms_iou_threshold",
        type=float,
        help="The threshold for non-maximum suppression",
        required=False,
        default=0.0,
    )

    args = parser.parse_args()
    assert (
        args.show_labels or args.show_preds
    ), "You must show either labels or predictions"
    if args.checkpoint_signature is None:
        print("Not loading checkpoint. Model will be randomly initialized.")
    visualize(args)
