import pandas as pd

from fpn.data import CustomVOCDetectionDataset
from fpn.evaluation.metrics import YOLOAccuracyMetric, YOLOBackgroundMetric, YOLOLocalizationMetric, YOLOOtherMetric
from fpn.models.model import Model


class ModelEvaluation:
    def __init__(self):
        self._results = []
        self.models = []

        self.metrics = [
            YOLOAccuracyMetric(),
            YOLOBackgroundMetric(),
            YOLOLocalizationMetric(),
            YOLOOtherMetric(),
        ]

    def evaluate_model(self, model: Model, dataset: CustomVOCDetectionDataset) -> None:
        """Given a model and a dataset, evaluate the model on the dataset on all metrics.

        Args:
            model (Model): The model to evaluate
            dataset (fpn.data.Dataset): The dataset to evaluate the model on
        """
        label = dataset.labels
        pred = model.predict(dataset)

        self.models.append(str(model))
        model_results = [metric.compute_value(pred, label) for metric in self.metrics]
        self._results.append(model_results)

    @property
    def results(self) -> pd.DataFrame:
        """Return the results of the evaluation as a pandas DataFrame and writes the df to evaluation.csv

        Returns:
            pd.DataFrame: the dataframe containing the results of the evaluation
                Index: the name of the model
                Columns: the metrics
        """

        df = pd.DataFrame(self._results, index=self.models, columns=[metric.name for metric in self.metrics])
        df.to_csv("evaluation.csv")
        return df
