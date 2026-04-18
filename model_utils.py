from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of: mean, sum, none")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        loss = (1.0 - pt).pow(self.gamma) * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    denom = m.sum(dim=1).clamp(min=1.0)
    return (x * m).sum(dim=1) / denom

def dense_nodes(x_node: torch.Tensor, batch_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return to_dense_batch(x_node, batch_vec)


@torch.no_grad()
def prepare_go_kv(
    go_graph_data: dict,
    go_num_terms: int,
    domain_dim: int,
    label_encoder,
    go_embed_weight: torch.Tensor,
    go_k_proj,
    go_v_proj,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_index = go_graph_data["edge_index"].to(device=device)
    if "biobert" in go_graph_data:
        go_x = go_graph_data["biobert"].to(device=device, dtype=dtype)
    else:
        go_x = go_embed_weight.to(device=device, dtype=dtype)
        if go_x.shape[0] != go_num_terms:
            raise ValueError(f"go_embed_weight rows({go_x.shape[0]}) != go_num_terms({go_num_terms})")

    go_z = label_encoder.encode(go_x, edge_index)
    if go_z.shape != (go_num_terms, domain_dim):
        raise ValueError(f"go_z.shape={tuple(go_z.shape)} expected ({go_num_terms},{domain_dim})")

    K = go_k_proj(go_z)
    V = go_v_proj(go_z)
    return K, V
