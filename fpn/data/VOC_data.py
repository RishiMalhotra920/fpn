import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.datasets import VOCDetection
from torchvision.transforms import v2 as transforms_v2


class CustomVOCDetectionDataset(VOCDetection):
    def __init__(
        self,
        root,
        image_set: str,
        transform: transforms_v2.Compose,
    ):
        assert image_set in ["train", "val"]
        super().__init__(root, year="2012", image_set=image_set, transform=None, target_transform=None)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """get the item at the index

        Args:
            index (int): index of the item

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict]: returns the image, label and metadata
                Image: torch.Tensor of shape (3, H, W)
                Label: torch.Tensor of shape (num_bboxes, 6) where 6 is [x1, y1, x2, y2, confidence class]
                Metadata: dict containing metadata: image_id, image_width, image_height, image_path
        """
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

        labels = []
        for object in objects:
            # normalize the bounding box coordinates
            x_min, x_max = int(object["bndbox"]["xmin"]), int(object["bndbox"]["xmax"])
            y_min, y_max = int(object["bndbox"]["ymin"]), int(object["bndbox"]["ymax"])

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(VOC_class_to_index[object["name"]])

        #  missing type annotations here.
        boxes = tv_tensors.BoundingBoxes(data=boxes, format="XYXY", canvas_size=(image_height, image_width))  # type: ignore

        out_image, label = self.transform(image, boxes)

        label = torch.cat([label.tensor, torch.tensor([1.0]), torch.tensor(labels).unsqueeze(1)], dim=1)

        return out_image, label, metadata

    def get_dataloader(self, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
        """Gets the dataloader for the dataset"""
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    @property
    def labels(self) -> list[np.ndarray]:
        """Return the labels of the dataset.

        Returns:
            list[np.ndarray]: ret[i] represents the bounding boxes in image i.
                Shape of ret[i]: nx6 where n is the number of bounding boxes in image i
        """
        _labels = []
        for i in range(len(self)):
            _, label, _ = self[i]
            _labels.append(label.numpy())

        return _labels

    @property
    def features(self):
        pass


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

VOC_class_to_index = {cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES)}
VOC_index_to_class = {idx: cls_name for idx, cls_name in enumerate(VOC_CLASSES)}
