from typing import Any

import torch
import yaml
from src.utils import class_to_index
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.datasets import VOCDetection
from torchvision.transforms import v2 as transforms_v2

config = yaml.safe_load(open("config.yaml"))


class CustomVOCDetection(VOCDetection):
    def __init__(
        self,
        root,
        year: str,
        image_set: str,
        transform: transforms_v2.Compose,
    ):
        super().__init__(
            root, year="2012", image_set="train", transform=None, target_transform=None
        )
        self.transform = transform

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        image, annotation = super().__getitem__(index)

        # transformed_image,  = self.transform(image)

        objects = annotation["annotation"]["object"]
        image_width = int(annotation["annotation"]["size"]["width"])
        image_height = int(annotation["annotation"]["size"]["height"])

        metadata = {
            "image_id": index,
            "image_width": image_width,
            "image_height": image_height,
            "image_path": annotation["annotation"]["filename"],
        }

        boxes = []

        # create bboxes and images of the format
        # [image, bbox] where bbox is of format [xmin, ymin, w, h]

        labels = []
        for object in objects:
            # normalize the bounding box coordinates
            x_min, x_max = int(object["bndbox"]["xmin"]), int(object["bndbox"]["xmax"])
            y_min, y_max = int(object["bndbox"]["ymin"]), int(object["bndbox"]["ymax"])

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_to_index[object["name"]])

        #  missing type annotations here.
        boxes = tv_tensors.BoundingBoxes(
            data=boxes, format="XYXY", canvas_size=(image_height, image_width)
        )  # type: ignore

        out_image, out_boxes = self.transform(image, boxes)

        out_yolo_grid = categorize_bboxes_into_grid(
            out_boxes, torch.tensor(labels), image_width, image_height
        )

        return out_image, out_yolo_grid, metadata

    @property
    def labels(self):
        # return the labels of the dataset as they should be used in ModelEvaluation
        out_image, out_boxes = self.transform(image, boxes)
        return out_boxes

    @property
    def features(self):
        pass


def create_datasets(
    root_dir: str, transform: transforms_v2.Compose
) -> tuple[Dataset, Dataset]:
    train_dataset = CustomVOCDetection(
        root=root_dir,
        year="2012",
        image_set="train",
        transform=transform,
        # download=False,
        # target_transform=target_transform,
    )

    val_dataset = CustomVOCDetection(
        root=root_dir,
        year="2012",
        image_set="val",
        transform=transform,
        # download=False,
        # target_transform=target_transform,
    )
    return train_dataset, val_dataset


def create_dataloaders(
    root_dir: str,
    transform: transforms_v2.Compose,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = create_datasets(root_dir, transform)

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_data_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return train_data_loader, val_data_loader
