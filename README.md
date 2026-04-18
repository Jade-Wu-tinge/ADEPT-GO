# ADEPTGO

ADEPTGO is a protein function prediction training pipeline based on graph features and GO-aware modeling.

## Setup

```bash
conda env create -f environment.yml
conda activate adeptgo
```

## Training

Run with config:

```bash
python train.py --config config.yaml
```

Optional CLI overrides:

```bash
python train.py --config config.yaml --ont bp --tag exp1 --ckpt_dir ./checkpoints
```

## Configuration

Main sections in `config.yaml`:

- `run`: seed, device, ontology, checkpoint naming
- `paths`: dataset and GO resource paths
- `training`: batch size, learning rate, epochs, data loader flags
- `model`: model dimensions and GO attention settings
- `bucket`: frequency bucket split thresholds
- `loss`: focal/bucketed loss settings

## Outputs

The best checkpoint is saved to:

- `{ckpt_dir}/best_model_{ont}_{graph_stem}_{tag}.pt` (if `tag` is set)
- `{ckpt_dir}/best_model_{ont}_{graph_stem}.pt` (if `tag` is empty)

## Notes

- Ensure all dataset paths in `config.yaml` exist before training.
- `train.py` now reads hyperparameters and paths from `config.yaml` by default.
