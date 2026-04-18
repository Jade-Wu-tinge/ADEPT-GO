from __future__ import annotations
from typing import Optional, Sequence, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

from evaluate import evaluate_cafa_short
from model_utils import FocalLoss
from model_go import GOMemoryBlock

class ResidualGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)

        self.conv1 = GCNConv(self.in_dim, self.hidden_dim, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim, add_self_loops=True, normalize=True)

        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)

        self.drop = nn.Dropout(dropout)
        self.res_proj = nn.Linear(self.in_dim, self.hidden_dim, bias=False)

    def forward(self, edge_index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pre = self.res_proj(x)
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = self.drop(F.relu(h))
        h = pre + h

        pre = h
        h2 = self.conv2(h, edge_index)
        h2 = self.bn2(h2)
        h2 = self.drop(F.relu(h2))
        h = pre + h2
        return h


class MultiModel(nn.Module):
    def __init__(
        self,
        num_labels: int,
        seq_dim: int = 1280,
        gcn_hidden: int = 256,
        go_graph_path: str = None,
        go_num_terms: Optional[int] = None,
        go_key_padding_mask: Optional[torch.Tensor] = None,
        token_level_go_q: bool = True,
        go_attn_heads: int = 8,
        attn_dropout: float = 0.1,
        mlp_hidden: int = 512,
        dropout: float = 0.1,
        use_freq_bucket_heads: bool = False,
        freq_bucket_indices: Optional[Mapping[str, Sequence[int]]] = None,
    ):
        super().__init__()
        self.num_labels = int(num_labels)
        self.seq_dim = int(seq_dim)

        self.hidden_dim = int(gcn_hidden)

        self.esm_proj = nn.Sequential(
            nn.Linear(self.seq_dim, self.seq_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gcn = ResidualGCN(in_dim=self.seq_dim, hidden_dim=self.hidden_dim, dropout=0.3)

        if go_graph_path is None:
            raise ValueError("go_graph_path is required")
        self.go_graph_data = torch.load(go_graph_path, map_location="cpu")
        if "edge_index_is_a" not in self.go_graph_data:
            raise ValueError("graph file must contain edge_index_is_a")
        if "edge_index_part_of" not in self.go_graph_data:
            raise ValueError("graph file must contain edge_index_part_of")
        if "biobert" not in self.go_graph_data:
            raise ValueError("graph file must contain biobert")
        if go_num_terms is None:
            go_num_terms = int(num_labels)
        self.go_num_terms = int(go_num_terms)
        self.go_key_padding_mask = go_key_padding_mask
        self.token_level_go_q = bool(token_level_go_q)

        self.go_block = GOMemoryBlock(
            go_graph_data=self.go_graph_data,
            go_num_terms=self.go_num_terms,
            seq_dim=self.seq_dim,
            hidden_dim=self.hidden_dim,
            go_attn_heads=go_attn_heads,
            attn_dropout=attn_dropout,
            token_level_go_q=self.token_level_go_q,
                )
        fused_dim = self.seq_dim + self.hidden_dim + self.seq_dim

        self.use_freq_bucket_heads = bool(use_freq_bucket_heads)
        self.freq_bucket_names = ("high", "mid", "low")

        if not self.use_freq_bucket_heads:
            self.mlp_all = self._make_bucket_mlp(
                in_dim=fused_dim,
                hidden_dim=mlp_hidden,
                out_dim=self.num_labels,
                dropout=dropout,
            )
        else:
            if not freq_bucket_indices:
                raise ValueError("freq_bucket_indices is required when use_freq_bucket_heads=True")

            high_idx = torch.tensor(list(freq_bucket_indices.get("high", [])), dtype=torch.long)
            mid_idx  = torch.tensor(list(freq_bucket_indices.get("mid", [])), dtype=torch.long)
            low_idx  = torch.tensor(list(freq_bucket_indices.get("low", [])), dtype=torch.long)

            self.register_buffer("high_idx", high_idx)
            self.register_buffer("mid_idx", mid_idx)
            self.register_buffer("low_idx", low_idx)

            cat_idx = torch.cat([x for x in [high_idx, mid_idx, low_idx] if x.numel() > 0], dim=0)
            if cat_idx.numel() == 0:
                raise ValueError("freq_bucket_indices cannot be empty")
            uniq_sorted = torch.unique(cat_idx).sort().values
            expected = torch.arange(self.num_labels, dtype=torch.long)
            if uniq_sorted.numel() != self.num_labels or not torch.equal(uniq_sorted, expected):
                raise ValueError("freq_bucket_indices must exactly cover [0, num_labels)")

            self.mlp_high = self._make_bucket_mlp(
                in_dim=fused_dim,
                hidden_dim=mlp_hidden,
                out_dim=int(self.high_idx.numel()),
                dropout=dropout,
            )
            self.mlp_mid = self._make_bucket_mlp(
                in_dim=fused_dim,
                hidden_dim=mlp_hidden,
                out_dim=int(self.mid_idx.numel()),
                dropout=dropout,
            )
            self.mlp_low = self._make_bucket_mlp(
                in_dim=fused_dim,
                hidden_dim=mlp_hidden,
                out_dim=int(self.low_idx.numel()),
                dropout=dropout,
            )

    def _make_bucket_mlp(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, out_dim),
        )


    def forward(self, batch):
        device = batch.x.device

        x = batch.x
        edge_index = batch.edge_index
        batch_vec = batch.batch

        x_proj = self.esm_proj(x)
        esm_pooled = global_mean_pool(x_proj, batch_vec)

        h_node = self.gcn(edge_index, x_proj)
        gcn_pooled = global_mean_pool(h_node, batch_vec)

        x_dense, mask_tok = to_dense_batch(x_proj, batch_vec)
        go_key_padding_mask = self.go_key_padding_mask
        if go_key_padding_mask is not None:
            go_key_padding_mask = go_key_padding_mask.to(device=device)

        go_after_struct = self.go_block(
            x_dense=x_dense,
            mask_tok=mask_tok,
            esm_pooled=esm_pooled,
            h_node=h_node,
            batch_vec=batch_vec,
            go_key_padding_mask=go_key_padding_mask,
        )
                
        feat = torch.cat([esm_pooled, gcn_pooled, go_after_struct], dim=-1)

        if not self.use_freq_bucket_heads:
            return self.mlp_all(feat)

        B = feat.size(0)
        logits = torch.zeros(B, self.num_labels, device=feat.device, dtype=feat.dtype)

        if self.high_idx.numel() > 0:
            logits[:, self.high_idx] = self.mlp_high(feat)
        if self.mid_idx.numel() > 0:
            logits[:, self.mid_idx] = self.mlp_mid(feat)
        if self.low_idx.numel() > 0:
            logits[:, self.low_idx] = self.mlp_low(feat)

        return logits

class MultiModelTrainer:
    def __init__(
        self,
        model: MultiModel,
        go_terms: Optional[Sequence[str]] = None,
        go_graph=None,
        ont: str = "mf",
        lr: float = 1e-3,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        use_bucketed_loss: bool = False,
        bucket_loss_weights: Optional[Mapping[str, float]] = None,
    ):
        self.model = model
        self.go_terms = list(go_terms) if go_terms is not None else None
        self.go_graph = go_graph
        self.ont = ont
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha) if use_focal_loss else nn.BCEWithLogitsLoss()
        self.use_bucketed_loss = bool(use_bucketed_loss)
        default_weights = {"high": 1.0, "mid": 1.2, "low": 1.6}
        self.bucket_loss_weights = dict(default_weights if bucket_loss_weights is None else bucket_loss_weights)

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.use_bucketed_loss:
            return self.criterion(logits, y)

        if not getattr(self.model, "use_freq_bucket_heads", False):
            return self.criterion(logits, y)

        weighted_loss = logits.new_tensor(0.0)
        weight_sum = 0.0
        for bucket_name in ("high", "mid", "low"):
            idx = getattr(self.model, f"{bucket_name}_idx", None)
            if idx is None or idx.numel() == 0:
                continue
            # Normalize per-bucket weight by bucket size.
            bucket_w = float(self.bucket_loss_weights.get(bucket_name, 1.0)) * float(idx.numel())
            weighted_loss = weighted_loss + bucket_w * self.criterion(logits[:, idx], y[:, idx])
            weight_sum += bucket_w

        if weight_sum == 0.0:
            return self.criterion(logits, y)
        return weighted_loss / weight_sum

    def train_one_epoch(self, loader, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.train().to(device)

        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            y = batch.y.float()
            logits = self.model(batch)
            loss = self._compute_loss(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total += float(loss.item()) * y.size(0)

        return total / len(loader.dataset)
        
    @torch.no_grad()
    def evaluate(self, loader, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval().to(device)

        total = 0.0
        ys, ps = [], []
        for batch in loader:
            batch = batch.to(device)
            y = batch.y.float()
            logits = self.model(batch)
            loss = self._compute_loss(logits, y)

            total += float(loss.item()) * y.size(0)
            p = torch.sigmoid(logits)
            ys.append(y.detach().cpu().numpy())
            ps.append(p.detach().cpu().numpy())

        avg_loss = total / len(loader.dataset)
        y_true = np.vstack(ys)
        y_score = np.vstack(ps)

        if self.go_terms is None or self.go_graph is None:
            raise ValueError("evaluate_cafa_short requires go_terms and go_graph")
        metrics = evaluate_cafa_short(
            y_true_np=y_true,
            y_score_np=y_score,
            goterms=self.go_terms,
            go_graph=self.go_graph,
            ont=self.ont,
        )
        macro_aupr = metrics["AUPR_macro"]
        micro_aupr = metrics["AUPR_micro"]
        fmax = metrics["Fmax"]
        return avg_loss, macro_aupr, micro_aupr, fmax, None, None, None
