from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch

from model_utils import masked_mean


class GOEncoderDenseSharedBase(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.drop = nn.Dropout(dropout)

        self.register_buffer("_Ahat_is_a", None, persistent=False)
        self.register_buffer("_Ahat_part", None, persistent=False)

    def _build_Ahat(self, edge_index: torch.Tensor, num_nodes: int, device, dtype) -> torch.Tensor:
        A = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)
        src, dst = edge_index[0], edge_index[1]
        A[src, dst] = 1.0

        A.fill_diagonal_(1.0)

        deg = A.sum(dim=1)
        deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
        Ahat = deg_inv_sqrt.unsqueeze(1) * A * deg_inv_sqrt.unsqueeze(0)
        return Ahat

    def _encode_with_Ahat(self, x: torch.Tensor, Ahat: torch.Tensor) -> torch.Tensor:
        h = Ahat @ x
        h = self.lin1(h)
        h = F.relu(h)
        h = self.drop(h)

        h = Ahat @ h
        h = self.lin2(h)
        return h

class DualGOEncoderDenseShared(nn.Module):
    def __init__(self, num_nodes: int, in_dim=768, out_dim=1280, dropout=0.1, use_input_residual=True):
        super(DualGOEncoderDenseShared, self).__init__()
        self.num_nodes = num_nodes
        
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        self.alpha_logit = nn.Parameter(torch.zeros(num_nodes, 1))

        self.part_specific_proj = nn.Linear(out_dim, out_dim)

        self.use_input_residual = use_input_residual
        if use_input_residual:
            self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            
        self._last_avg_alpha = 0.5

    def forward(self, x, Ahat_is_a, Ahat_part):
        base = self.dropout(self.act(self.lin1(x)))
        base = self.lin2(base) 

        h_is_a = torch.matmul(Ahat_is_a, base)
        h_part = torch.matmul(Ahat_part, base)

        h_part = self.part_specific_proj(h_part)

        if self.use_input_residual:
            r = self.res_proj(x)
            h_is_a = h_is_a + r
            h_part = h_part + r

        alpha = torch.sigmoid(self.alpha_logit)
        h_go = alpha * h_is_a + (1.0 - alpha) * h_part

        self._last_avg_alpha = alpha.detach().mean().item()

        return h_go, {"alpha": alpha}


class GOMemoryBlock(nn.Module):
    def __init__(
        self,
        self_graph_dummy=None,
        *,
        go_graph_data,
        go_num_terms: int,
        seq_dim: int = 1280,
        hidden_dim: int = 256,
        go_attn_heads: int = 8,
        attn_dropout: float = 0.1,
        token_level_go_q: bool = False,
    ):
        super().__init__()
        self.go_graph_data = go_graph_data
        self.go_num_terms = int(go_num_terms)
        self.go_dim = int(seq_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_level_go_q = bool(token_level_go_q)

        self.esm_to_go_q = nn.Linear(self.go_dim, self.go_dim)

        go_in_dim = int(self.go_graph_data["biobert"].shape[1])
        self.go_label_encoder = DualGOEncoderDenseShared(
            num_nodes=self.go_num_terms,
            in_dim=go_in_dim,
            out_dim=self.go_dim,
            dropout=0.1,
            use_input_residual=True,
        )

        self.go_k_proj = nn.Linear(self.go_dim, self.go_dim)
        self.go_v_proj = nn.Linear(self.go_dim, self.go_dim)

        self.go_attn = nn.MultiheadAttention(
            embed_dim=self.go_dim, num_heads=go_attn_heads, dropout=attn_dropout, batch_first=True
        )
        self.go_attn_ln = nn.LayerNorm(self.go_dim)
        self.go2struct_attn = nn.MultiheadAttention(
            embed_dim=self.go_dim, num_heads=go_attn_heads, dropout=attn_dropout,
            kdim=self.hidden_dim, vdim=self.hidden_dim, batch_first=True
        )
        self.go2struct_ln = nn.LayerNorm(self.go_dim)

        self._cached_k = None
        self._cached_v = None
        self._cached_device = None
        self._cached_dtype = None
        self._last_go_alpha = 0.5
        
        self.register_buffer("_Ahat_is_a", None, persistent=False)
        self.register_buffer("_Ahat_part", None, persistent=False)

    def clear_go_cache(self):
        self._cached_k = None
        self._cached_v = None
        self._cached_device = None
        self._cached_dtype = None

    def _build_ahat(self, edge_index, n, device, dtype):
        mat = torch.eye(n, device=device, dtype=dtype)
        mat[edge_index[0], edge_index[1]] = 1.0
        deg = mat.sum(dim=1).clamp(min=1e-12)
        d_inv_sqrt = deg.pow(-0.5)
        return d_inv_sqrt.unsqueeze(1) * mat * d_inv_sqrt.unsqueeze(0)

    def _compute_go_kv(self, device, dtype):
        if self._Ahat_is_a is None or self._Ahat_is_a.device != device:
            self._Ahat_is_a = self._build_ahat(self.go_graph_data["edge_index_is_a"], self.go_num_terms, device, dtype)
            self._Ahat_part = self._build_ahat(self.go_graph_data["edge_index_part_of"], self.go_num_terms, device, dtype)

        go_x = self.go_graph_data["biobert"].to(device=device, dtype=dtype)

        go_z, aux = self.go_label_encoder(go_x, self._Ahat_is_a, self._Ahat_part)

        self._last_go_alpha = aux["alpha"].detach().mean().item()

        K = self.go_k_proj(go_z)
        V = self.go_v_proj(go_z)
        return K, V

    def forward(
        self,
        *,
        x_dense: torch.Tensor,
        mask_tok: torch.Tensor,
        esm_pooled: torch.Tensor,
        h_node: torch.Tensor,
        batch_vec: torch.Tensor,
        go_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        device = esm_pooled.device
        dtype = esm_pooled.dtype
        B = int(esm_pooled.size(0))

        if self.token_level_go_q:
            q_go = self.esm_to_go_q(x_dense)
        else:
            q_go = self.esm_to_go_q(esm_pooled).unsqueeze(1)

        if self.training:
            K, V = self._compute_go_kv(device, dtype)
        else:
            need = (
                (self._cached_k is None)
                or (self._cached_device != device)
                or (self._cached_dtype != dtype)
            )
            if need:
                with torch.no_grad():
                    K, V = self._compute_go_kv(device, dtype)
                self._cached_k = K
                self._cached_v = V
                self._cached_device = device
                self._cached_dtype = dtype

            K = self._cached_k
            V = self._cached_v

        T = int(K.size(0))

        Kb = K.unsqueeze(0).expand(B, T, self.go_dim)
        Vb = V.unsqueeze(0).expand(B, T, self.go_dim)

        kpm = None
        if go_key_padding_mask is not None:
            kpm = go_key_padding_mask.to(device=device).unsqueeze(0).expand(B, T)

        go_out, _ = self.go_attn(q_go, Kb, Vb, key_padding_mask=kpm, need_weights=False)

        if self.token_level_go_q:
            go_read = masked_mean(go_out, mask_tok)
        else:
            go_read = go_out.squeeze(1)

        go_vec = self.go_attn_ln(go_read)

        kv_struct, mask_struct = to_dense_batch(h_node, batch_vec)
        key_padding_mask = ~mask_struct

        q_struct = go_vec.unsqueeze(1)
        attn_out2, _ = self.go2struct_attn(
            q_struct,
            kv_struct,
            kv_struct,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        go_after_struct = self.go2struct_ln(go_vec + attn_out2.squeeze(1))
        return go_after_struct