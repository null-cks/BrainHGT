import torch
import pandas as pd


def calculate_shortest_path_distances(adj: torch.Tensor, max_value: float = 1e6) -> torch.Tensor:
    """
    Computes the all-pairs shortest path distance matrix for a batch of weighted adjacency matrices
    using the Floyd-Warshall algorithm.
    """
    device = adj.device
    batch_size, num_roi, _ = adj.shape

    eye = torch.eye(num_roi, device=device).unsqueeze(0)
    mask = eye.bool()

    D = adj.clone()
    D = torch.where(mask, torch.zeros_like(D), torch.where(D == 0, max_value, D))

    # Floyd-Warshall algorithm
    for k in range(num_roi):
        D_ik = D[:, :, k].unsqueeze(-1)
        D_kj = D[:, k, :].unsqueeze(1)
        D_new = D_ik + D_kj
        D = torch.minimum(D, D_new)

    valid_mask = D < max_value
    batch_max = torch.where(valid_mask, D, -torch.inf).amax(dim=(1, 2), keepdim=True)

    batch_max = torch.clamp_min(batch_max, 0.0)

    D = torch.where(valid_mask, D, batch_max + 1)

    D = torch.where(mask, torch.zeros_like(D), D)

    return D


def get_dice(file_path):

    dice = pd.read_csv(file_path)

    return torch.tensor(dice.to_numpy()).float()
