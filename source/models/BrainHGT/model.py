import torch
import torch.nn as nn
from omegaconf import DictConfig
from .components.attention import LSRATransformerEncoder
from .components.clustering import PriorGuidedClustering
from .utils import calculate_shortest_path_distances, get_dice


class BrainHGT(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        num_nodes = cfg.dataset.node_sz
        node_feature_sz = cfg.dataset.node_feature_sz
        dim_hidden = cfg.model.dim_hidden
        num_heads = cfg.model.num_heads
        num_layers = cfg.model.num_layers
        dropout = cfg.model.dropout
        dim_feedforward = cfg.model.dim_feedforward
        num_clusters = cfg.model.num_communities
        encoder_hidden_dim = cfg.model.encoder_hidden_dim
        num_class = cfg.dataset.num_class
        dice_prior = get_dice(cfg.model.dice_path)

        self.embedding = nn.Linear(in_features=node_feature_sz, out_features=dim_hidden)

        self.lsra_layers = nn.ModuleList([
            LSRATransformerEncoder(d_model=dim_hidden, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.clustering_module = PriorGuidedClustering(
            cluster_number=num_clusters,
            embedding_dimension=dim_hidden,
            node_number=num_nodes,
            dice_prior=dice_prior,
            encoder_hidden_dimension=encoder_hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(dim_hidden, 32),
            nn.LeakyReLU(),
            nn.Linear(32, num_class)
        )

    def forward(self, time_series, node_feature: torch.Tensor, adj: torch.Tensor):

        adj = (adj > 0).float()

        shp = calculate_shortest_path_distances(adj)

        node_embed = self.embedding(node_feature)

        # Node-level Feature Learning (LSRA)
        node_interaction = None
        for lsra_layer in self.lsra_layers:
            node_embed, node_interaction = lsra_layer(node_embed, sph=shp)

        # Community-level Feature Learning
        community_features, assignment, community_interaction = self.clustering_module(node_embed)

        # Classification
        graph_representation = torch.mean(community_features, dim=1)
        output = self.classifier(graph_representation)

        return output, node_interaction, assignment, community_interaction