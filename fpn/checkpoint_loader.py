import os
import shutil
from pathlib import Path

import neptune
import torch
import yaml

config = yaml.safe_load(open("config.yaml"))


def download_checkpoint(run_id: str, checkpoint_path: str) -> tuple[Path, str]:
    """
    Downloads a checkpoint from Neptune.

    Args:
        run_id: The Neptune run id.
        checkpoint_path: The checkpoint path.

    Example usage:
        download_checkpoint(run_id="IM-23", checkpoint_path="checkpoints/epoch_10.pth")
    """

    # save to checkpoints/{run_id}/file_name.pth
    checkpoints_dir = Path(f"checkpoints/{run_id}")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run = neptune.init_run(
        project="towards-hi/image-classification",
        with_id=run_id,
        mode="read-only",
        api_token=config["neptune_api_token"],
    )
    run[checkpoint_path].download(destination=str(checkpoints_dir))

    # load from temp_dir/{file_name}.pth into model
    file_name = checkpoint_path.split("/")[-1]

    return checkpoints_dir, file_name


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

    Neptune stores checkpoints in the format IM-23:checkpoints/epoch_10.pth.
    Locally, we store it as checkpoints/IM-23/epoch_10.pth.

    I should probably migrate to b2 at some point.

    Args:
        model: A target PyTorch model to load.
        epoch: The epoch number to load the model from.

    Example usage:
        load_model(model=model_0, epoch=5)
    """
    assert (
        ":" in checkpoint_signature
    ), "checkpoint_signature should be in the format RunId:CheckpointPath"

    try:
        run_id, checkpoint_path = checkpoint_signature.split(":")
        assert not checkpoint_path.endswith(
            ".pth"
        ), "checkpoint_path should not end with .pth"

        file_name = checkpoint_path.split("/")[-1]
        if os.path.exists(f"checkpoints/{run_id}/{file_name}.pth"):
            checkpoints_dir = Path(f"checkpoints/{run_id}")
        else:
            checkpoints_dir, file_name = download_checkpoint(run_id, checkpoint_path)

        params = torch.load(
            checkpoints_dir / f"{file_name}.pth", map_location=torch.device("cpu")
        )
        model.load_state_dict(params)
        # file_name should be epoch_{epoch}.pth
        epoch_number = int(file_name.split("_")[1])

        # we save logs for the epoch number that was completed
        # we should start logging from the next epoch
        start_epoch = epoch_number + 1

        print("this is epoch_number", epoch_number)

    except KeyboardInterrupt:
        print("Interrupted loading checkpoint, removing partially loaded checkpoint")
        # delete_directory(checkpoints_dir)
        os.remove(f"checkpoints/{run_id}/{file_name}.pth")
        print("Removed partially loaded checkpoint")
        raise KeyboardInterrupt

    return start_epoch


def load_checkpoint_for_yolo_from_pretrained_image_net_model(
    model: torch.nn.Module, checkpoint_signature: str
) -> None:
    """
    the weights loaded in are for nn.DataParallel(FeatureExtractor -> YOLOPretrainNet)
    the model in parameters is nn.DataParallel(FeatureExtractor -> YOLONet)
    load the weights for Feature Extractor from pretrained model and random initialize the rest
    """
    run_id, checkpoint_path = checkpoint_signature.split(":")
    assert not checkpoint_path.endswith(
        ".pth"
    ), "checkpoint_path should not end with .pth"

    temp_dir, file_name = download_checkpoint(run_id, checkpoint_path)
    params = torch.load(temp_dir / f"{file_name}.pth", map_location=torch.device("cpu"))
    model_dict = model.state_dict()
    # model.load_state_dict(params)
    # 1. filter out keys in pretrained model not in current model
    pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
    # print(
    # f"image net model keys: {params.keys()}\n yolo model keys: {model_dict.keys()}\n keys loading: {pretrained_dict.keys()}"
    # )
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print(f"[INFO] Loaded model from {checkpoint_signature}")

    # delete_directory(temp_dir)
