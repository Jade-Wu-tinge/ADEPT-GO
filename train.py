import os
import torch
import obonet
import argparse
from pathlib import Path
from typing import Any
import yaml

from utils import build_all_loaders
from model import MultiModel, MultiModelTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ont", type=str, default=None, choices=["mf", "bp", "cc"])
    parser.add_argument("--go_graph_path", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError("Config file must be a mapping at top level")
    return cfg


def cfg_get(cfg: dict[str, Any], path: str, default=None):
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_cc_frequency_buckets(train_loader, num_labels: int, q_low=0.25, q_high=0.75):
    cnt = torch.zeros(num_labels, dtype=torch.float64)
    with torch.no_grad():
        for batch in train_loader:
            y = batch.y
            if y.dim() != 2 or y.size(1) != num_labels:
                raise ValueError(f"Invalid batch.y shape: expected [B,{num_labels}], got {tuple(y.shape)}")
            cnt += y.float().sum(dim=0).cpu().to(torch.float64)

    sorted_idx = torch.argsort(cnt, descending=False)
    n_labels = int(num_labels)

    low_k = max(1, int(round(n_labels * float(q_low))))
    high_k = max(1, int(round(n_labels * float(1.0 - q_high))))
    if low_k + high_k > n_labels:
        high_k = max(1, n_labels - low_k)

    low_idx = sorted_idx[:low_k]
    high_idx = sorted_idx[n_labels - high_k:]
    mid_idx = sorted_idx[low_k:n_labels - high_k]

    bucket_indices = {
        "high": high_idx.tolist(),
        "mid": mid_idx.tolist(),
        "low": low_idx.tolist(),
    }
    return cnt, bucket_indices


def verify_bucket_coverage(bucket_indices, num_labels: int):
    high = set(bucket_indices["high"])
    mid = set(bucket_indices["mid"])
    low = set(bucket_indices["low"])
    all_idx = high | mid | low
    overlap = (high & mid) | (high & low) | (mid & low)
    missing = set(range(num_labels)) - all_idx
    extra = all_idx - set(range(num_labels))
    ok = len(overlap) == 0 and len(missing) == 0 and len(extra) == 0
    return ok, overlap, missing, extra


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device_cfg = str(cfg_get(cfg, "run.device", "auto")).lower()
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    processed_dir = cfg_get(cfg, "paths.processed_dir", "/root/autodl-tmp/datasets/processed")
    annot_path = cfg_get(cfg, "paths.annot_path", "/root/autodl-tmp/datasets/labels/nrPDB-GO_2019.06.18_annot.tsv")
    annot_path_af2 = cfg_get(cfg, "paths.annot_path_af2", "/root/autodl-tmp/datasets/labels/nrSwiss-Model-GO_annot.tsv")
    go_obo_path = cfg_get(cfg, "paths.go_obo_path", "/root/autodl-tmp/datasets/labels/go-basic.obo")

    batch_size = int(cfg_get(cfg, "training.batch_size", 16))
    lr = float(cfg_get(cfg, "training.lr", 3e-5))
    epochs = int(cfg_get(cfg, "training.epochs", 100))
    early_stop_patience = int(cfg_get(cfg, "training.early_stop_patience", 5))
    num_workers = int(cfg_get(cfg, "training.num_workers", 0))
    keep_fp16 = bool(cfg_get(cfg, "training.keep_fp16", False))
    use_af2 = bool(cfg_get(cfg, "training.use_af2", True))

    ont = args.ont or cfg_get(cfg, "run.ont", "mf")
    go_graph_path = args.go_graph_path or cfg_get(
        cfg, "run.go_graph_path", "/root/autodl-tmp/code/DO_preprocess/双图/noclosure/mf_dual_bidirectional_graph.pt"
    )
    tag = args.tag if args.tag is not None else cfg_get(cfg, "run.tag", "")
    ckpt_dir = args.ckpt_dir or cfg_get(cfg, "run.ckpt_dir", ".")

    use_freq_bucket_heads = bool(cfg_get(cfg, "model.use_freq_bucket_heads", True))
    token_level_go_q = bool(cfg_get(cfg, "model.token_level_go_q", True))
    seq_dim = int(cfg_get(cfg, "model.seq_dim", 1280))
    gcn_hidden = int(cfg_get(cfg, "model.gcn_hidden", 256))
    go_attn_heads = int(cfg_get(cfg, "model.go_attn_heads", 8))

    use_bucketed_loss = bool(cfg_get(cfg, "loss.use_bucketed_loss", True))
    use_focal_loss = bool(cfg_get(cfg, "loss.use_focal_loss", True))
    focal_gamma = float(cfg_get(cfg, "loss.focal_gamma", 2.0))
    bucket_loss_weights = cfg_get(cfg, "loss.bucket_loss_weights", {"high": 1.0, "mid": 1.2, "low": 1.6})

    q_low = float(cfg_get(cfg, "bucket.q_low", 0.60))
    q_high = float(cfg_get(cfg, "bucket.q_high", 0.90))

    train_loader, val_loader, test_loader, meta = build_all_loaders(
        processed_dir=processed_dir,
        annot_path=annot_path,
        annot_path_af2=annot_path_af2,
        ont=ont,
        batch_size=batch_size,
        num_workers=num_workers,
        keep_fp16=keep_fp16,
        use_af2=use_af2,
    )

    num_labels = meta["num_labels"]
    go_terms = meta["go_terms"][ont]
    if len(go_terms) != num_labels:
        raise ValueError("go_terms and num_labels mismatch; label order cannot be guaranteed")

    go_graph = obonet.read_obo(open(go_obo_path, "r"))

    _, freq_bucket_indices = build_cc_frequency_buckets(
        train_loader=train_loader,
        num_labels=num_labels,
        q_low=q_low,
        q_high=q_high,
    )
    cov_ok, overlap, missing, extra = verify_bucket_coverage(freq_bucket_indices, num_labels)
    if not cov_ok:
        raise ValueError(
            f"Bucket coverage validation failed: overlap={len(overlap)} missing={len(missing)} extra={len(extra)}"
        )

    graph_stem = Path(go_graph_path).stem
    run_name = f"{ont}_{graph_stem}"
    if tag:
        run_name = f"{run_name}_{tag}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"best_model_{run_name}.pt")

    model_kwargs = dict(
        num_labels=num_labels,
        seq_dim=seq_dim,
        gcn_hidden=gcn_hidden,
        go_graph_path=go_graph_path,
        go_num_terms=num_labels,
        token_level_go_q=token_level_go_q,
        go_attn_heads=go_attn_heads,
        use_freq_bucket_heads=use_freq_bucket_heads,
        freq_bucket_indices=freq_bucket_indices if use_freq_bucket_heads else None,
    )

    try:
        model = MultiModel(**model_kwargs).to(device)
    except RuntimeError as e:
        if device == "cuda" and "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            device = "cpu"
            model = MultiModel(**model_kwargs).to(device)
        else:
            raise

    trainer = MultiModelTrainer(
        model=model,
        go_terms=go_terms,
        go_graph=go_graph,
        ont=ont,
        lr=lr,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        use_bucketed_loss=use_bucketed_loss,
        bucket_loss_weights=bucket_loss_weights,
    )

    best_fmax = -1.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_one_epoch(train_loader, device=device)
        model.go_block.clear_go_cache()

        val_loss, macro_aupr, micro_aupr, fmax, _, _, _ = trainer.evaluate(
            val_loader, device=device
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"micro_AUPR={micro_aupr:.4f} | "
            f"Fmax={fmax:.4f}"
        )

        if fmax > best_fmax:
            best_fmax = fmax
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "fmax": fmax,
                    "freq_bucket_indices": freq_bucket_indices,
                    "q_low": q_low,
                    "q_high": q_high,
                    "go_terms": go_terms,
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                break

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.go_block.clear_go_cache()

    test_loss, macro_aupr, micro_aupr, fmax, _, _, _ = trainer.evaluate(
        test_loader, device=device
    )

    print(
        f"[TEST] loss={test_loss:.4f} | "
        f"micro_AUPR={micro_aupr:.4f} | "
        f"Fmax={fmax:.4f}"
    )


if __name__ == "__main__":
    main()