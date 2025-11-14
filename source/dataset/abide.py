import numpy as np
import torch
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_abide_data(cfg: DictConfig):
    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    final_timeseries = data["timeseries"]
    final_pearson = data["corr"]
    labels = data["label"]
    adj = data["omst"]
    site = data['site']

    scaler = StandardScaler(mean=np.mean(
        final_timeseries), std=np.std(final_timeseries))

    final_timeseries = scaler.transform(final_timeseries)

    final_timeseries, final_pearson, adj, labels = [torch.from_numpy(
        data).float() for data in (final_timeseries, final_pearson, adj, labels)]

    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]

    return final_timeseries, final_pearson, adj, labels.squeeze(), site
