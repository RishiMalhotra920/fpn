import torch


def _get_anchor_positions(anchor_widths: torch.Tensor, anchor_heights: torch.Tensor, s: int, image_dim: int, device: str) -> torch.Tensor:
    """
    Get anchor positions for the volume in the shape:
    (1, feature_map_height*feature_map_width*anchor_heights*anchor_widths, 4)
    """
    x_step = image_dim / s
    y_step = image_dim / s

    grid = torch.zeros(s, s, len(anchor_heights), 4, device=device)

    x_grid_cell_centers = ((torch.arange(0, s, device=device).float() * x_step) + (x_step / 2)).reshape(1, s, 1)
    y_grid_cell_centers = ((torch.arange(0, s, device=device).float() * y_step) + (y_step / 2)).reshape(s, 1, 1)

    anchor_widths_grid = anchor_widths.reshape(1, 1, len(anchor_widths))
    anchor_heights_grid = anchor_heights.reshape(1, 1, len(anchor_heights))

    # x and y centers broadcast from ( s, 1, 1) to (s, s, 9)
    # widths and heights broadcast from ( 1, 1, 9) to ( s, s, 9)
    grid[:, :, :, 0] = x_grid_cell_centers - anchor_widths_grid / 2  # (1, s, s, 9)
    grid[:, :, :, 1] = y_grid_cell_centers - anchor_heights_grid / 2
    grid[:, :, :, 2] = x_grid_cell_centers + anchor_widths_grid / 2
    grid[:, :, :, 3] = y_grid_cell_centers + anchor_heights_grid / 2

    reshaped_grid = grid.reshape(s * s * len(anchor_heights), 4)

    return reshaped_grid  # (s*s*9, 4)


def create_anchors(image_size: tuple[int, int], device: str) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    fpn_map_small_anchor_scales = torch.tensor([32.0, 64.0, 128.0], device=device)
    fpn_map_medium_anchor_scales = torch.tensor([64.0, 128.0, 256.0], device=device)
    fpn_map_large_anchor_scales = torch.tensor([128.0, 256.0, 512.0], device=device)
    anchor_ratios = torch.tensor([0.5, 1, 2], device=device)
    all_anchor_scales = [
        fpn_map_small_anchor_scales,
        fpn_map_medium_anchor_scales,
        fpn_map_large_anchor_scales,
    ]

    all_anchor_widths = []  # list([(9, ), (9, ), (9, )])
    all_anchor_heights = []  # list([(9, ), (9, ), (9, )])
    all_anchor_positions = []
    feature_map_dims = [100, 50, 25]  # found through experimentation
    for anchor_scales, s in zip(all_anchor_scales, feature_map_dims):
        permutations = torch.cartesian_prod(anchor_scales, anchor_ratios)
        widths = permutations[:, 0] * permutations[:, 1]  # (9, )
        heights = permutations[:, 0] * (1 / permutations[:, 1])  # (9, )

        anchor_positions = _get_anchor_positions(widths, heights, s, image_size[0], device=device)

        all_anchor_widths.append(widths)
        all_anchor_heights.append(heights)
        all_anchor_positions.append(anchor_positions)

    return all_anchor_widths, all_anchor_heights, all_anchor_positions
