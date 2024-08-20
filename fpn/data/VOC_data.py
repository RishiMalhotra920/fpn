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
        self.max_num_bbox = 60

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """get the item at the index

        Args:
            index (int): index of the item

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict]: returns the image, label and metadata
                Image: torch.Tensor of shape (3, H, W)
                Cls: torch.Tensor of shape (num_bbox) where each element is the class index
                Boxes: torch.Tensor of shape (num_bbox, 4) where 4 is [x1, y1, x2, y2]
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

        boxes_list = []

        cls_list = []
        for object in objects:
            # normalize the bounding box coordinates
            x_min, x_max = int(object["bndbox"]["xmin"]), int(object["bndbox"]["xmax"])
            y_min, y_max = int(object["bndbox"]["ymin"]), int(object["bndbox"]["ymax"])

            boxes_list.append([x_min, y_min, x_max, y_max])
            cls_list.append(VOC_class_to_index[object["name"]])

        tv_bbox = tv_tensors.BoundingBoxes(data=boxes_list, format="XYXY", canvas_size=(image_height, image_width))  # type: ignore

        out_image, bbox = self.transform(image, tv_bbox)

        cls = torch.tensor(cls_list)

        padded_bbox = torch.zeros((self.max_num_bbox, 4))
        padded_cls = torch.zeros(self.max_num_bbox, dtype=torch.long)
        num_gt_bbox_in_each_image = bbox.shape[0]
        padded_cls[:num_gt_bbox_in_each_image] = cls
        padded_bbox[:num_gt_bbox_in_each_image] = bbox

        return out_image, padded_cls, padded_bbox, num_gt_bbox_in_each_image, metadata  # type: ignore

    def get_dataloader(self, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
        """Gets the dataloader for the dataset"""
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


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
    "background",
]
BACKGROUND_CLASS_INDEX = VOC_CLASSES.index("background")

VOC_class_to_index = {cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES)}
VOC_index_to_class = {idx: cls_name for idx, cls_name in enumerate(VOC_CLASSES)}
