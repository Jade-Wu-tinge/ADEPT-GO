import csv
import os
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_annot(annot_path: str):
    onts = ["mf", "bp", "cc"]
    prot2annot = {}
    go_terms = {ont: [] for ont in onts}
    go_names = {ont: [] for ont in onts}

    with open(annot_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")

        next(reader); go_terms["mf"] = next(reader)
        next(reader); go_names["mf"] = next(reader)

        next(reader); go_terms["bp"] = next(reader)
        next(reader); go_names["bp"] = next(reader)

        next(reader); go_terms["cc"] = next(reader)
        next(reader); go_names["cc"] = next(reader)

        next(reader)
        go_counts = {ont: np.zeros(len(go_terms[ont]), dtype=np.float32) for ont in onts}
        term2idx = {ont: {t: i for i, t in enumerate(go_terms[ont])} for ont in onts}

        for row in reader:
            pid = row[0]
            go = row[1:]
            prot2annot[pid] = {}

            for i, ont in enumerate(onts):
                terms = [t for t in go[i].split(",") if t.strip()]
                one_hot = np.zeros(len(go_terms[ont]), dtype=np.float32)
                for term in terms:
                    j = term2idx[ont].get(term)
                    if j is not None:
                        one_hot[j] = 1.0
                        go_counts[ont][j] += 1.0
                prot2annot[pid][ont] = one_hot

    return prot2annot, go_terms, go_names, go_counts


def _merge_annots(primary, extra):
    prot2annot_1, go_terms_1, go_names_1, go_counts_1 = primary
    prot2annot_2, go_terms_2, go_names_2, go_counts_2 = extra

    for ont in ("mf", "bp", "cc"):
        if go_terms_1[ont] != go_terms_2[ont]:
            raise ValueError(f"GO term lists do not match for {ont}; cannot merge annotations")

    prot2annot = dict(prot2annot_1)
    for pid, annot in prot2annot_2.items():
        prot2annot.setdefault(pid, annot)

    go_counts = {ont: (go_counts_1[ont] + go_counts_2[ont]) for ont in ("mf", "bp", "cc")}
    return prot2annot, go_terms_1, go_names_1, go_counts

def load_ids_from_pdbch_pt(pdbch_pt_path: str, split: str) -> list[str]:
    key_map = {"train": "train_pdbch", "val": "val_pdbch", "test": "test_pdbch"}
    if split not in key_map:
        raise ValueError(f"split must be train/val/test, got {split}")
    key = key_map[split]

    obj = torch.load(pdbch_pt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"{pdbch_pt_path} must be a dict, got {type(obj)}")
    if key not in obj:
        raise KeyError(f"{pdbch_pt_path} is missing key='{key}', available keys={list(obj.keys())}")

    ids = obj[key]
    if not isinstance(ids, list) or (len(ids) > 0 and not isinstance(ids[0], str)):
        raise TypeError(f"{pdbch_pt_path}[{key}] must be list[str], got {type(ids)}")

    ids = [s.strip() for s in ids if s.strip()]
    if not ids:
        raise ValueError(f"{pdbch_pt_path}[{key}] is empty or all whitespace")
    return ids

def load_graphs_old(graph_pt_path: str):
    graphs = torch.load(graph_pt_path, map_location="cpu")
    if not isinstance(graphs, list) or len(graphs) == 0:
        raise TypeError(f"{graph_pt_path} must be a non-empty list[Data], got {type(graphs)}")
    return graphs


def old_get(g, field: str):
    d = getattr(g, "__dict__", None)
    if d is None or field not in d:
        raise KeyError(f"Field '{field}' not found in legacy Data object. Example fields: {sorted(list(d.keys()))[:30]} ...")
    return d[field]


def load_esm2_dict(esm2_pt_path: str) -> dict:
    if not os.path.exists(esm2_pt_path):
        raise FileNotFoundError(f"ESM2 embedding file not found: {esm2_pt_path}")

    obj = torch.load(esm2_pt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"{esm2_pt_path} must be dict[pid]->Tensor, got {type(obj)}")

    out = {}
    bad = []
    for k, v in obj.items():
        if not isinstance(k, str):
            bad.append((type(k), type(v)))
            continue
        if not torch.is_tensor(v):
            bad.append((type(k), type(v)))
            continue
        out[k.strip()] = v

    if bad:
        raise TypeError(
            f"{esm2_pt_path} contains invalid entries (key must be str, value must be Tensor), "
            f"examples: {bad[:3]}"
        )
    if not out:
        raise ValueError(f"{esm2_pt_path} is empty and cannot provide ESM2 features")
    return out


def merge_esm2_dicts(primary: dict, extra: dict) -> dict:
    merged = dict(primary)
    for pid, x in extra.items():
        if pid in merged:
            x0 = merged[pid]
            if tuple(x0.shape) != tuple(x.shape):
                raise ValueError(
                    f"Duplicate ESM2 pid with inconsistent shapes: {pid} {tuple(x0.shape)} vs {tuple(x.shape)}"
                )
            continue
        merged[pid] = x
    return merged


def _drop_missing_esm2_samples(ids: list[str], graphs_old: list, esm2_dict: dict, split: str):
    if len(ids) != len(graphs_old):
        raise ValueError(f"{split}: len(ids)={len(ids)} != len(graphs)={len(graphs_old)}")

    keep_idx = [i for i, pid in enumerate(ids) if pid in esm2_dict]

    if not keep_idx:
        raise RuntimeError(f"{split}: all samples are missing ESM2 embeddings; cannot build DataLoader")

    ids_kept = [ids[i] for i in keep_idx]
    graphs_kept = [graphs_old[i] for i in keep_idx]
    return ids_kept, graphs_kept

class MAEFGraphAnnotDataset(Dataset):
    def __init__(
        self,
        graphs_old,
        ids: list[str],
        prot2annot: dict,
        ont: str = "mf",
        keep_fp16: bool = True,
    ):
        if ont not in ("mf", "bp", "cc"):
            raise ValueError(f"ont must be mf/bp/cc, got {ont}")

        if len(graphs_old) != len(ids):
            raise ValueError(f"Alignment failed: len(graphs_old)={len(graphs_old)} != len(ids)={len(ids)}")

        missing = [pid for pid in ids if pid not in prot2annot]
        if missing:
            raise KeyError(f"prot2annot is missing {len(missing)} IDs (example: {missing[:5]})")

        self.graphs_old = graphs_old
        self.ids = ids
        self.prot2annot = prot2annot
        self.ont = ont
        self.keep_fp16 = keep_fp16

        for i in (0, len(ids)//2, len(ids)-1):
            g = graphs_old[i]
            _ = old_get(g, "x")
            _ = old_get(g, "edge_index")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Data:
        pid = self.ids[idx]
        g_old = self.graphs_old[idx]

        x = old_get(g_old, "x")
        if (not self.keep_fp16) and x.dtype != torch.float32:
            x = x.float()

        edge_index = old_get(g_old, "edge_index").long()

        y_np = self.prot2annot[pid][self.ont]
        y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(0)

        d = Data(x=x, edge_index=edge_index, y=y)
        d.pid = pid
        return d

def build_split_loader(
    graph_pt_path: str,
    pdbch_pt_path: str,
    split: str,
    prot2annot: dict,
    ont: str = "mf",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    keep_fp16: bool = True,
    drop_last: bool = False,
):
    ids = load_ids_from_pdbch_pt(pdbch_pt_path, split=split)
    graphs_old = load_graphs_old(graph_pt_path)
    if len(ids) != len(graphs_old):
        raise ValueError(f"{split}: len(ids)={len(ids)} != len(graphs)={len(graphs_old)}")

    ds = MAEFGraphAnnotDataset(graphs_old, ids, prot2annot, ont=ont, keep_fp16=keep_fp16)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    # Ensure labels are batched as [B, T].
    b = next(iter(loader))
    if b.y.dim() != 2:
        raise RuntimeError(
            f"{split}: invalid batch.y shape {tuple(b.y.shape)}, expected [B,T].\n"
            "This usually means y was not stacked per graph correctly."
        )
    return loader, ds


def build_split_loader_multi(
    graph_pt_paths: list[str],
    pdbch_pt_paths: list[str],
    split: str,
    prot2annot: dict,
    ont: str = "mf",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    keep_fp16: bool = True,
    drop_last: bool = False,
):
    if len(graph_pt_paths) != len(pdbch_pt_paths):
        raise ValueError("graph_pt_paths and pdbch_pt_paths must have the same length")

    ids_all = []
    graphs_all = []
    for g_path, p_path in zip(graph_pt_paths, pdbch_pt_paths):
        ids = load_ids_from_pdbch_pt(p_path, split=split)
        graphs_old = load_graphs_old(g_path)
        if len(ids) != len(graphs_old):
            raise ValueError(f"{split}: len(ids)={len(ids)} != len(graphs)={len(graphs_old)}")
        ids_all.extend(ids)
        graphs_all.extend(graphs_old)

    ds = MAEFGraphAnnotDataset(graphs_all, ids_all, prot2annot, ont=ont, keep_fp16=keep_fp16)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    b = next(iter(loader))
    if b.y.dim() != 2:
        raise RuntimeError(
            f"{split}: invalid batch.y shape {tuple(b.y.shape)}, expected [B,T].\n"
            "This usually means y was not stacked per graph correctly."
        )
    return loader, ds


def build_all_loaders(
    processed_dir: str,
    annot_path: str,
    ont: str = "mf",
    batch_size: int = 8,
    num_workers: int = 0,
    keep_fp16: bool = True,
    use_af2: bool = False,
    annot_path_af2: Optional[str] = None,
):
    primary = load_annot(annot_path)
    if use_af2:
        if not annot_path_af2:
            raise ValueError("annot_path_af2 is required when use_af2=True")
        extra = load_annot(annot_path_af2)
        prot2annot, go_terms, go_names, go_counts = _merge_annots(primary, extra)

        train_graphs = [f"{processed_dir}/train_graph.pt", f"{processed_dir}/AF2train_graph.pt"]
        train_pdbch = [f"{processed_dir}/train_pdbch.pt", f"{processed_dir}/AF2train_pdbch.pt"]
        train_loader, train_ds = build_split_loader_multi(
            graph_pt_paths=train_graphs,
            pdbch_pt_paths=train_pdbch,
            split="train",
            prot2annot=prot2annot,
            ont=ont,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            keep_fp16=keep_fp16,
            drop_last=True,
        )
        val_loader, val_ds = build_split_loader(
            graph_pt_path=f"{processed_dir}/val_graph.pt",
            pdbch_pt_path=f"{processed_dir}/val_pdbch.pt",
            split="val",
            prot2annot=prot2annot,
            ont=ont,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            keep_fp16=keep_fp16,
        )
        test_loader, test_ds = build_split_loader(
            graph_pt_path=f"{processed_dir}/test_graph.pt",
            pdbch_pt_path=f"{processed_dir}/test_pdbch.pt",
            split="test",
            prot2annot=prot2annot,
            ont=ont,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            keep_fp16=keep_fp16,
        )
    else:
        prot2annot, go_terms, go_names, go_counts = primary
        train_loader, train_ds = build_split_loader(
            graph_pt_path=f"{processed_dir}/train_graph.pt",
            pdbch_pt_path=f"{processed_dir}/train_pdbch.pt",
            split="train",
            prot2annot=prot2annot,
            ont=ont,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            keep_fp16=keep_fp16,
            drop_last=True,
        )
        val_loader, val_ds = build_split_loader(
            graph_pt_path=f"{processed_dir}/val_graph.pt",
            pdbch_pt_path=f"{processed_dir}/val_pdbch.pt",
            split="val",
            prot2annot=prot2annot,
            ont=ont,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            keep_fp16=keep_fp16,
        )
        test_loader, test_ds = build_split_loader(
            graph_pt_path=f"{processed_dir}/test_graph.pt",
            pdbch_pt_path=f"{processed_dir}/test_pdbch.pt",
            split="test",
            prot2annot=prot2annot,
            ont=ont,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            keep_fp16=keep_fp16,
        )

    meta = {
        "prot2annot": prot2annot,
        "go_terms": go_terms,
        "go_names": go_names,
        "go_counts": go_counts,
        "num_labels": len(go_terms[ont]),
    }

    return train_loader, val_loader, test_loader, meta





    
