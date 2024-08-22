import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


def denormalize_image(image: torch.Tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize an image and convert it from 0..1 to 0..255 range using PyTorch transforms.

    Args:
    image (torch.Tensor): Image tensor of shape (C, H, W) in 0..1 range
    mean (list): Mean used for normalization
    std (list): Standard deviation used for normalization

    Returns:
    torch.Tensor: Denormalized image in 0..255 range
    """
    denormalize = transforms.Compose(
        [
            transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),
        ]
    )

    return denormalize(image)


def visualize_bbox(
    images: torch.Tensor,
    pred_labels: list[torch.Tensor],
    # pred_confidences: list[torch.Tensor],
    pred_bbox: list[torch.Tensor],
    target_labels: list[torch.Tensor],
    target_bbox: list[torch.Tensor],
    should_denormalize: bool = True,
    show_pred: bool = True,
    show_gt: bool = True,
):
    num_images = images.shape[0]
    fig, axes = plt.subplots(1, num_images, figsize=(15 * num_images, 15))

    for i in range(num_images):
        image = images[i]
        if should_denormalize:
            image = denormalize_image(image)

        ax = axes[i]

        ax.imshow(image.permute(1, 2, 0))

        if show_gt:
            image_target_bbox = target_bbox[i]
            image_target_labels = target_labels[i]

            for bbox_idx in range(len(image_target_bbox)):
                x1, y1, x2, y2 = image_target_bbox[bbox_idx].tolist()
                label = image_target_labels[bbox_idx].item()
                width = x2 - x1
                height = y2 - y1

                rect = patches.Rectangle(
                    (x1, y1),
                    width,
                    height,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )

                ax.add_patch(rect)
                ax.text(
                    x1,
                    y1 - 10,
                    f"Label: {label}",
                    color="white",
                    fontsize=12,
                    bbox=dict(facecolor="red", alpha=0.5),
                )

            if show_pred:
                image_pred_bbox = pred_bbox[i]
                image_pred_labels = pred_labels[i]

                for bbox_idx in range(len(image_pred_bbox)):
                    x1, y1, x2, y2 = image_pred_bbox[bbox_idx].tolist()
                    label = image_pred_labels[bbox_idx].item()
                    width = x2 - x1
                    height = y2 - y1

                    rect = patches.Rectangle(
                        (x1, y1),
                        width,
                        height,
                        linewidth=2,
                        edgecolor="r",
                        facecolor="none",
                    )

                    ax.add_patch(rect)
                    ax.text(
                        x1,
                        y1 - 10,
                        f"Pred: {label}",
                        color="white",
                        fontsize=12,
                        bbox=dict(facecolor="red", alpha=0.5),
                    )

        ax.axis("off")

    plt.show()
