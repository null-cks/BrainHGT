import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainDecayAttention(nn.Module):
    """
        Core attention mechanism with topological decay for short-range heads.
    """
    def __init__(self, num_heads, init_hop=2.0, gamma=1.0, learnable_hop=True, learnable_gamma=True):
        super().__init__()
        assert num_heads % 2 == 0, "Number of heads must be even for long/short split."
        self.num_heads = num_heads
        self.half_heads = num_heads // 2

        # Learnable parameters for the short-range heads
        if learnable_hop:
            self.short_hop = nn.Parameter(torch.full((self.half_heads,), float(init_hop)))
        else:
            self.register_buffer('short_hop', torch.full((self.half_heads,), float(init_hop)))

        if learnable_gamma:
            self.short_gamma = nn.Parameter(torch.ones(self.half_heads) * gamma)
        else:
            self.register_buffer('short_gamma', torch.ones(self.half_heads) * gamma)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sph: torch.Tensor):
        # Split heads for long-range and short-range attention
        q_short, q_long = torch.split(q, [self.half_heads, self.half_heads], dim=1)
        k_short, k_long = torch.split(k, [self.half_heads, self.half_heads], dim=1)
        v_short, v_long = torch.split(v, [self.half_heads, self.half_heads], dim=1)

        # --- Short-Range Attention with Topological Decay ---
        attn_short_raw = torch.matmul(q_short, k_short.transpose(-1, -2)) / math.sqrt(q_short.size(-1))

        # Expand shortest-path-length matrix and hop parameters for broadcasting
        sph_expanded = sph.unsqueeze(1)  # [B, 1, N, N]
        hop_expanded = self.short_hop.view(1, self.half_heads, 1, 1)  # [1, H/2, 1, 1]

        # Calculate decay mask based on distance from hop threshold
        decay_factors = F.relu(sph_expanded - hop_expanded)
        # mask = F.leaky_relu(sph_expanded - hop_expanded)

        # Apply sigmoid to gamma to constrain it to (0, 1)
        gamma = torch.sigmoid(self.short_gamma).view(1, self.half_heads, 1, 1)

        decay_mask = torch.pow(gamma, decay_factors)

        attn_short_decayed = attn_short_raw * decay_mask
        attn_weights_short = F.softmax(attn_short_decayed, dim=-1)
        output_short = torch.matmul(attn_weights_short, v_short)

        # --- Long-Range Attention ---
        attn_long_raw = torch.matmul(q_long, k_long.transpose(-1, -2)) / math.sqrt(q_long.size(-1))
        attn_weights_long = F.softmax(attn_long_raw, dim=-1)
        output_long = torch.matmul(attn_weights_long, v_long)

        combined_output = torch.cat([output_short, output_long], dim=1)
        combined_weights = torch.cat([attn_weights_short, attn_weights_long], dim=1)

        return combined_output, combined_weights


class MultiHeadAttentionWrapper(nn.Module):
    """A standard multi-head attention wrapper."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = BrainDecayAttention(num_heads)

    def forward(self, q, k, v, sph):
        batch_size = q.size(0)

        # Project and reshape for multi-head attention
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply attention
        attn_output, attn_weights = self.attention(q, k, v, sph=sph)

        # Reshape and project back to output space
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class LSRATransformerEncoder(nn.Module):
    """
    A single layer of the Long-Short Range Attention Transformer.
    This follows the standard Transformer encoder architecture.
    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionWrapper(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, sph):
        # Self-Attention Block
        src2, attn_weights = self.self_attn(src, src, src, sph=sph)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-Forward Block
        src2 = self.linear2(self.dropout(F.leaky_relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, attn_weights