from abc import abstractmethod

import numpy as np
from torch.utils.data import Dataset

from fpn.models.model import Model


class FasterRCNN_FPN(Model):
    def __init__(self, nms_threshold: float):
        super().__init__(nms_threshold)

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def predict(self, dataset: Dataset) -> list[np.ndarray]:
        """Given a dataset, predict the output of the model on the dataset.

        Args:
            dataset (Dataset): A dataset object.

        Returns:
            np.ndarray: A numpy array of predictions
                Shape: list[nx6]. ret[i] represents bounding boxes in image i.
                Shape of ret[i]: nx6 where n is the number of bounding boxes in image i
                and 6 is the number of values in the bounding box: x1, y1, x2, y2, confidence, class of the bounding
        """
        pass
