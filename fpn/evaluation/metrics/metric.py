from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    @abstractmethod
    def compute_value(self, pred: list[np.ndarray], gt: list[np.ndarray]) -> float:
        """Computes the value of the metric

        Args:
            pred: list of np arrays of predicted values
            gt: list of np arrays of ground truth values
            **kwargs: Additional keyword arguments that may be required by specific metric implementations


        Returns:
            The value of the metric
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Gets the name of the metric

        Returns:
            str: Name of the metric
        """
        pass

    @abstractmethod
    def is_larger_better(self) -> bool:
        """Determines if a larger value of the metric is better

        Returns:
            True if a larger value of the metric is better, False otherwise
        """
        pass
