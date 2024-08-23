import glob
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import neptune
import torch
import yaml

config = yaml.safe_load(open("config.yaml"))


def assert_filepaths_exist(base_path, file_patterns):
    for pattern in file_patterns:
        full_pattern = os.path.join(base_path, pattern)
        matched_files = glob.glob(full_pattern)
        assert len(matched_files) > 0, f"No files found for pattern: {pattern}"
        for file in matched_files:
            assert os.path.exists(file), f"File does not exist: {file}"


class RunManager:
    """
    The job of the run manager is to manage experiment runs. It integrates
    """

    def __init__(
        self,
        *,
        new_run_name: str | None = None,
        load_from_run_id: str | None = None,
        tags: list[str] = [],
        source_files: list[str] = [],
    ):
        # assert new_run_name is not None, "new_run_name should not be None"

        if new_run_name is not None:
            base_path = os.getcwd()
            assert_filepaths_exist(base_path, source_files)

            self.temp_dir = Path("temp")
            self.temp_dir.mkdir(exist_ok=True)

            self.run = neptune.init_run(
                project="towards-hi/fpn-with-faster-rcnn-object-detection",
                api_token=config["neptune_api_token"],
                name=new_run_name,
                source_files=source_files,
                tags=tags,
            )
        else:
            self.run = None

        # if new run is none, don't do anything - for testing purposes!

    def add_tags(self, tags: list[str]) -> None:
        """
        Add tags to the run.

        Args:
          tags: a list of tags to add to the run.

        Example:
          tags = ["resnet", "cifar10"]
        """
        if self.run is None:
            return

        self.run["sys/tags"].add(tags)

    def set_checkpoint_to_continue_from(self, checkpoint_to_continue_from_signature: str) -> None:
        """
        Set the checkpoint to continue from.

        Args:
          checkpoint_to_continue_from_signature: a string in the format RunId:CheckpointPath
        """
        if self.run is None:
            return

        self.log_data({"checkpoint_to_continue_from_signature": checkpoint_to_continue_from_signature})

    def log_data(self, data: Dict[str, Any]) -> None:
        """
        Log data to the run.

        Args:
          data: a dictionary of data to log.

        Example:
          data = {
            "num_epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 32,
            "hidden_units": 512,
            "loss_fn": "CrossEntropyLoss",
            "optimizer": "Adam",
            "device": "cuda"
          }
        """
        if self.run is None:
            return

        for key in data:
            self.run[key] = data[key]

    def log_filesets(self, fileset_map: Dict[str, list[str]]) -> None:
        """
        Log filesets to the run.

        Args:
          filesets: a dictionary of filesets to log.

        Example:
          filesets = {
            "model": ["model_builder.py", "model_trainer.py"],
            "data": ["data_loader.py", "models/*.py", "data_loaders"]
          }
          you can use wildcards to upload all files in a directory
          or directory names!
        """
        if self.run is None:
            return

        for fileset_name in fileset_map:
            self.run[fileset_name].upload_files(fileset_map[fileset_name])

    def log_files(self, files: Dict[str, str]) -> None:
        """
        Log files to the run.

        Args:
          files: a dictionary of files to log.

        Example:
          files = {
            "model/code": "model_builder.py"
          }
        """
        if self.run is None:
            return

        for key in files:
            self.run[key].upload(files[key])

    def log_metrics(self, metrics: Dict[str, float], epoch: float) -> None:
        """
        Track metrics for the run and plot it on neptune.

        epoch can be a float - a float epoch denotes that we are logging a fraction of the way through the epoch

        Args:
          metrics: a dictionary of metrics to track.

        Example:
          metrics = {
            "train/loss": 0.5,
            "val/loss": 0.3,
            "train/accuracy": 0.8,
            "val/accuracy": 0.9
          }
        """
        if self.run is None:
            return

        for metric_name in metrics:
            self.run[metric_name].append(metrics[metric_name], step=epoch)
            # print(f"\nEpoch: {epoch}, {metric_name}: {metrics[metric_name]}")

    def end_run(self):
        if self.run is None:
            return

        self.run.stop()
        try:
            shutil.rmtree(self.temp_dir)

        except Exception as e:
            print(f"Failed to remove directory: {e}")

    def save_model(self, checkpoint_dir: str, model: torch.nn.Module, epoch: int) -> None:
        """Saves a PyTorch model to a target directory.

        Args:
          model: A target PyTorch model to save.
          epoch: The epoch number to save the model at.

        Example usage:
          save_model(model=model_0, epoch=5)
        """
        # Note that model should be saved as epoch_{epoch}.pth
        # to add more info, do this: epoch_{epoch}_lr_{lr}_bs_{bs}.pth
        # later on you can implement a json file to store info about checkpoints

        # need this here in case temp dir is deleted between run creation and model saving
        if self.run is None:
            return

        self.temp_dir.mkdir(exist_ok=True)
        model_save_path = self.temp_dir / checkpoint_dir / f"{epoch}.pth"
        print(f"[INFO] Saving model to {model_save_path}")
        torch.save(obj=model.state_dict(), f=model_save_path)
        self.run[f"{checkpoint_dir}/epoch_{epoch}"].upload(str(model_save_path))
