import pandas as pd


class ModelEvaluation:
    def __init__(self, datasets, metrics):
        self.results = []

    def evaluate_model(self, model, dataset):
        pass

    def get_results(self):
        return pd.DataFrame(self.results)
