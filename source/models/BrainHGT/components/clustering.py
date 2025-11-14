import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.functional import normalize
from entmax import entmax15


class PriorGuidedClustering(nn.Module):
    """
    Groups node features into functional communities using cross-attention
    guided by an anatomical prior (Dice score matrix). This module is the
    core of the "Prior-Guided Clustering Module" from the paper.
    """

    def __init__(
            self,
            cluster_number: int,
            embedding_dimension: int,
            node_number: int,
            dice_prior: torch.Tensor,
            encoder_hidden_dimension: int = 32,
            num_heads: int = 8,
            dropout: float = 0.1,
            orthogonal_init: bool = True,
            freeze_centers: bool = True,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.num_heads = num_heads

        assert embedding_dimension % num_heads == 0, "Embedding dimension must be divisible by num_heads"
        self.head_dim = embedding_dimension // num_heads

        self.register_buffer('dice_prior', normalize(dice_prior, p=2, dim=1))

        self.encoder = nn.Sequential(
            nn.Linear(embedding_dimension * node_number, encoder_hidden_dimension),
            nn.LeakyReLU(),
            nn.Linear(encoder_hidden_dimension, encoder_hidden_dimension),
            nn.LeakyReLU(),
            nn.Linear(encoder_hidden_dimension, embedding_dimension * node_number),
        )

        # Initialize community prototypes (cluster centers)
        initial_centers = torch.zeros(cluster_number, embedding_dimension)
        nn.init.xavier_uniform_(initial_centers)

        if orthogonal_init:
            with torch.no_grad():
                for i in range(1, cluster_number):
                    proj = torch.sum(initial_centers[:i] * initial_centers[i], dim=1, keepdim=True)
                    initial_centers[i] -= torch.sum(proj * initial_centers[:i], dim=0)
                    initial_centers[i] /= torch.norm(initial_centers[i], p=2)

        self.community_prototypes = Parameter(initial_centers, requires_grad=not freeze_centers)

        # Projection layers for cross-attention
        self.q_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.k_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.v_proj = nn.Linear(embedding_dimension, embedding_dimension)

        # Self-attention and FFN to refine community representations
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dimension, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension * 4),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dimension * 4, embedding_dimension)
        )

        # Normalization and Dropout layers
        self.norm1 = nn.LayerNorm(embedding_dimension)
        self.norm2 = nn.LayerNorm(embedding_dimension)
        self.norm3 = nn.LayerNorm(embedding_dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor):
        batch_size, node_num, _ = node_features.shape

        flattened_features = node_features.view(batch_size, -1)
        encoded_flat = self.encoder(flattened_features)
        refined_node_features = encoded_flat.view(batch_size, node_num, self.embedding_dimension)

        # --- 1. Anatomically Constrained Cross-Attention ---
        # Query: Community prototypes, Key/Value: Node features
        q_input = self.community_prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        q = self.q_proj(q_input).view(batch_size, self.cluster_number, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(refined_node_features).view(batch_size, node_num, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(refined_node_features).view(batch_size, node_num, self.num_heads, self.head_dim).transpose(1, 2)

        # Modulate attention with the Dice prior
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        dice_expanded = self.dice_prior.T.unsqueeze(0).unsqueeze(0)
        biased_attn_scores = attn_scores * dice_expanded

        # Use EntMax for sparse assignments
        attn_weights = entmax15(biased_attn_scores, dim=-1)
        soft_assignment = attn_weights.mean(dim=1).transpose(1, 2)

        cross_attn_output = torch.matmul(attn_weights, v)
        cross_attn_output = cross_attn_output.transpose(1, 2).contiguous().view(batch_size, self.cluster_number, self.embedding_dimension)
        community_features = self.norm1(q_input + self.dropout1(cross_attn_output))

        # --- 2. Refine Community Representations ---
        self_attn_output, community_interaction = self.self_attn(community_features, community_features, community_features)
        community_features = self.norm2(community_features + self.dropout2(self_attn_output))

        ffn_output = self.ffn(community_features)
        refined_community_features = self.norm3(community_features + self.dropout3(ffn_output))

        return refined_community_features, soft_assignment, community_interaction