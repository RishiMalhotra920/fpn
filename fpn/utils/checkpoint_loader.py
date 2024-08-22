import os
import shutil
from pathlib import Path

import neptune
import torch
import yaml

config = yaml.safe_load(open("config.yaml"))


def download_checkpoint(run_id: str, neptune_folder_name: str, neptune_file_name: str, disk_checkpoints_dir: Path, disk_file_name: str):
    """
    Downloads a checkpoint from Neptune.

    Args:
        run_id: The Neptune run id.
        checkpoint_path: The checkpoint path.

    Example usage:
        download_checkpoint(run_id="IM-23", checkpoint_path="checkpoints/epoch_10.pth")
    """

    # save to checkpoints/{run_id}/file_name.pth
    disk_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    neptune_checkpoint_path = f"{neptune_folder_name}/{neptune_file_name}"
    run = neptune.init_run(
        project="towards-hi/fpn-with-faster-rcnn-object-detection",
        with_id=run_id,
        mode="read-only",
        api_token=config["neptune_api_token"],
    )
    run[neptune_checkpoint_path].download(destination=str(disk_checkpoints_dir))


def delete_directory(temp_dir: Path) -> None:
    """
    Deletes a checkpoint from the file system.

    Args:
        run_id: The Neptune run id.
        checkpoint_path: The checkpoint path.

    Example usage:
        delete_checkpoint_from_file_system(run_id="IM-23", checkpoint_path="checkpoints/epoch_10.pth")
    """
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Failed to remove directory: {e}")


def load_checkpoint(model: torch.nn.Module, checkpoint_signature: str) -> int:
    """
    Loads a PyTorch model weights from a run at an epoch. If the checkpoint is not found locally,
    it is downloaded.

    """
    assert ":" in checkpoint_signature, "checkpoint_signature should be in the format RunId:CheckpointPath"

    try:
        run_id, checkpoint_path = checkpoint_signature.split(":")
        assert not checkpoint_path.endswith(".pth"), "checkpoint_path should not end with .pth"

        neptune_folder_name = "/".join(checkpoint_path.split("/")[:-1])
        neptune_file_name = checkpoint_path.split("/")[-1]

        disk_checkpoints_dir = Path(f"checkpoints/{run_id}/{neptune_folder_name}")
        disk_file_name = neptune_file_name
        if not os.path.exists(f"{disk_checkpoints_dir}/{disk_file_name}.pth"):
            download_checkpoint(run_id, neptune_folder_name, neptune_file_name, disk_checkpoints_dir, disk_file_name)

        params = torch.load(disk_checkpoints_dir / f"{disk_file_name}.pth", map_location=torch.device("cpu"))
        model.load_state_dict(params)
        # file_name should be epoch_{epoch}.pth
        epoch_number = int(disk_file_name.split("_")[1])

        # we save logs for the epoch number that was completed
        # we should start logging from the next epoch
        start_epoch = epoch_number + 1

        print("this is epoch_number", epoch_number)

    except KeyboardInterrupt:
        print("Interrupted loading checkpoint, removing partially loaded checkpoint")
        # delete_directory(checkpoints_dir)
        os.remove(f"checkpoints/{run_id}/{disk_file_name}.pth")
        print("Removed partially loaded checkpoint")
        raise KeyboardInterrupt

    return start_epoch
