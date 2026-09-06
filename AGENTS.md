# Repository Guidelines

Contributor guide for **LATR: 3D Lane Detection from Monocular Images with Transformer** (PyTorch, ICCV 2023). The model detects 3D lanes from monocular images on the OpenLane, ONCE, and Apollo datasets.

## Project Structure & Module Organization

- `main.py` - entry point; parses args, loads an mmcv `Config`, dispatches to `Runner.train()` or `Runner.eval()`.
- `models/` - network definition (`latr.py`, `latr_head*.py`, `embedding.py`, `transformer_bricks.py`, `sparse_ins*.py`).
- `data/` - dataset loaders and transforms (`Load_Data.py`, `apollo_dataset.py`, `transform.py`).
- `experiments/` - train/eval loop (`runner.py`), DDP setup (`ddp.py`), GPU helpers.
- `utils/` - 3D-lane evaluators (`eval_3D_lane*.py`, `eval_3D_once.py`) and shared utilities.
- `config/` - mmcv-style Python configs (`_base_/` base configs, `release_iccv/` per-dataset configs).
- `pretrained_models/` - checkpoint storage (download separately; gitignored except `.gitkeep`).
- `visulization/` - Flask visualization app (`app.py`).
- `docs/` - setup, data prep, and train/eval guides; `assets/` holds paper figures.
- `work_dirs/` - experiment outputs (gitignored).

## Build, Test, and Development Commands

Setup (see `docs/install.md`): install PyTorch 1.8.0 (CUDA 10.1), then `python -m pip install -r requirements.txt`, plus `mmcv==1.5.0`, `mmdet==2.24.0`, `mmdet3d==1.0.0rc3`.

Train a dataset:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 main.py --config config/release_iccv/latr_1000_baseline.py
```
Convenience wrappers: `train.sh`, `train_Apollo.sh`, `train_Once.sh`.

Evaluate by adding `--cfg-options evaluate=true eval_ckpt=pretrained_models/<ckpt>.pth` (see `test.sh`, `test_Apollo.sh`, `test_Once.sh`). Launch the visualization app with `python visulization/app.py`.

## Coding Style & Naming Conventions

- Python with `snake_case`; module configs are plain `.py` files using the mmcv `Config` system.
- No linter/formatter is configured - match surrounding style (4-space indent).
- The directory is spelled `visulization` throughout; keep the existing spelling rather than "fixing" it.

## Testing Guidelines

There is no unit-test suite. Correctness is verified by running evaluation through `main.py` (metrics: F1, accuracy, X/Z errors) via the scripts in `utils/eval_3D_lane*.py`. Reproduce a reported metric before submitting model changes.

## Commit & Pull Request Guidelines

Git history uses short, capitalized, present-tense summaries (e.g., `Add salience codes`, `Fix bugs`, `Update`). No conventional-commit prefixes; keep messages under ~60 characters.

For pull requests: describe the change and dataset affected, link related issues, and include before/after evaluation metrics. Do not commit `*.pth` checkpoints or `work_dirs/` (already in `.gitignore`).
