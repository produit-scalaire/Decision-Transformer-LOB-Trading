#!/usr/bin/env python3
"""
Full offline pipeline: data generation, training, evaluation, and a DeepLOB-style
context-length profile — without a full factorial that runs for days.

Design (defaults)
-----------------
- **6 000 train trajectories** per dataset (``generator.train_episodes``).
- **Two rewards:** ``mid_price``, ``shaped``.
- **Two state representations:** ``raw``, ``log_returns``.
- **Return-to-go horizons:** full-episode (``null``) for the 2×2 grid, plus
  ``H=50`` and ``H=100`` for ``mid_price`` + ``raw`` only (separate datasets).
- **Context length K:** all jobs above train at **K=100**, except one extra
  **K=50** run on the canonical ``mid_price`` / ``raw`` / full-horizon data for
  the horizon-vs-DeepLOB comparison plot.
- **CNN:** a single train+eval on canonical data (K=100).
- **Larger DT:** one run with ``d_model=256``, ``n_layers=4``, ``n_heads=8`` on
  the same canonical data; all other transformer jobs use default sizes from
  ``configs/config.yaml``.

Outputs under ``outputs/full_profile/<timestamp>/``: per-job model dirs, eval
plots, ``run_manifest.json``, ``aggregate_metrics.csv``, and
``context_horizon_profile.{csv,png}`` from the K=50 vs K=100 transformer pair.

Requires FI-2010 via kagglehub (same as ``main.py``). Use ``--dry-run`` to
print the plan only.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from omegaconf import DictConfig, OmegaConf

from src.data.trajectories_generator import generate_dataset_pipeline
from src.evaluations.dt_viz import _load_train_initial_rtg, evaluate_model
from src.training.training_pipeline import train_model


def _load_context_horizon_profile():
    path = _ROOT / "scripts" / "context_horizon_profile.py"
    spec = importlib.util.spec_from_file_location("context_horizon_profile", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_omega_config() -> DictConfig:
    cfg_path = _ROOT / "configs" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw.pop("defaults", None)
    return OmegaConf.create(raw)


def data_key(reward_type: str, state_representation: str, reward_horizon: int | None) -> str:
    h = "full" if reward_horizon is None else str(int(reward_horizon))
    r = "mp" if reward_type == "mid_price" else "sh"
    s = "raw" if state_representation == "raw" else "logrt"
    return f"{r}_{s}_H{h}"


@dataclass(frozen=True)
class DataSpec:
    reward_type: str
    state_representation: str
    reward_horizon: int | None

    @property
    def key(self) -> str:
        return data_key(self.reward_type, self.state_representation, self.reward_horizon)


def unique_data_specs() -> list[DataSpec]:
    grid = [
        DataSpec("mid_price", "raw", None),
        DataSpec("shaped", "raw", None),
        DataSpec("mid_price", "log_returns", None),
        DataSpec("shaped", "log_returns", None),
        DataSpec("mid_price", "raw", 50),
        DataSpec("mid_price", "raw", 100),
    ]
    return grid


def ensure_dataset(
    spec: DataSpec,
    run_root: Path,
    base_cfg: DictConfig,
    train_episodes: int,
    test_episodes: int,
    plot_subdir: str = "data_distributions",
) -> tuple[Path, Path]:
    """Generate train/test trajectories if missing."""
    data_root = run_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    train_path = data_root / f"{spec.key}_train.pt"
    test_path = data_root / f"{spec.key}_test.pt"
    if train_path.is_file() and test_path.is_file():
        print(f"[data] Reusing {spec.key}")
        return train_path, test_path

    reward_shaping = OmegaConf.to_container(base_cfg.generator.reward_shaping, resolve=True)
    dist_plot_dir = str(run_root / plot_subdir / spec.key)
    Path(dist_plot_dir).mkdir(parents=True, exist_ok=True)

    print(f"[data] Generating {spec.key} → {train_path.name}")
    generate_dataset_pipeline(
        train_episodes=train_episodes,
        test_episodes=test_episodes,
        workers=int(base_cfg.hardware.workers),
        train_out=str(train_path),
        test_out=str(test_path),
        plot_dir=dist_plot_dir,
        window_size=int(base_cfg.generator.window_size),
        episode_length=int(base_cfg.generator.episode_length),
        reward_type=spec.reward_type,
        reward_shaping=reward_shaping,
        state_representation=spec.state_representation,
        price_offset=float(base_cfg.generator.price_offset),
        reward_horizon=spec.reward_horizon,
    )
    return train_path, test_path


def merge_model_cfg(base: DictConfig, overrides: dict[str, Any]) -> DictConfig:
    m = OmegaConf.create(OmegaConf.to_container(base.model, resolve=True))
    for k, v in overrides.items():
        OmegaConf.update(m, k, v, merge=False)
    return m


def best_dt_row(df_metrics: pd.DataFrame) -> pd.Series:
    dt_idx = [i for i in df_metrics.index if str(i).startswith("DT ")]
    if not dt_idx:
        return pd.Series(dtype=float)
    sub = df_metrics.loc[dt_idx]
    if "Sharpe" not in sub.columns:
        return sub.iloc[0]
    sharpe = sub["Sharpe"]
    if sharpe.notna().any():
        return sub.loc[sharpe.idxmax()]
    return sub.iloc[0]


def train_with_batch_backoff(
    train_data_path: Path,
    model_dir: Path,
    plot_dir: Path,
    model_cfg: DictConfig,
    train_cfg: DictConfig,
    hardware_cfg: DictConfig,
    min_batch_size: int = 128,
) -> tuple[float, int]:
    """Train a model, halving batch size on CUDA OOM until it fits."""
    batch_size = int(train_cfg.batch_size)
    while True:
        local_train_cfg = OmegaConf.create(OmegaConf.to_container(train_cfg, resolve=True))
        local_train_cfg.batch_size = batch_size
        t0 = time.perf_counter()
        try:
            train_model(
                train_data_path=str(train_data_path),
                model_dir=str(model_dir),
                model_cfg=model_cfg,
                train_cfg=local_train_cfg,
                hardware_cfg=hardware_cfg,
                plot_dir=str(plot_dir),
            )
            return time.perf_counter() - t0, batch_size
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if batch_size <= min_batch_size:
                raise
            next_batch = max(min_batch_size, batch_size // 2)
            print(
                f"[train] CUDA OOM for batch_size={batch_size}. "
                f"Retrying with batch_size={next_batch}."
            )
            batch_size = next_batch


def run_context_profile(
    run_root: Path,
    ckpt_k50: Path,
    ckpt_k100: Path,
    train_data: Path,
    test_data: Path,
    model_cfg: dict[str, Any],
    state_representation: str,
    max_eval_trajectories: int,
    rtg_rollout_mode: str,
) -> pd.DataFrame:
    chp = _load_context_horizon_profile()
    rtgs = _load_train_initial_rtg(str(train_data))
    target_rtg = float(np.median(rtgs)) if len(rtgs) else 1.0
    plot_dir = run_root / "context_profile"
    plot_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {50: ckpt_k50, 100: ckpt_k100}
    return chp.run_profile(
        checkpoints=checkpoints,
        test_data=test_data,
        plot_dir=plot_dir,
        target_rtg=target_rtg,
        max_eval_trajectories=max_eval_trajectories,
        state_representation=state_representation,
        model_cfg=model_cfg,
        rtg_rollout_mode=rtg_rollout_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Full DT profile pipeline (bounded runtime).")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_ROOT / "outputs" / "full_profile",
        help="Base directory; a timestamped subfolder is created per run.",
    )
    parser.add_argument("--train-episodes", type=int, default=6000, help="Train trajectories per dataset.")
    parser.add_argument("--test-episodes", type=int, default=6000, help="Test trajectories per dataset.")
    parser.add_argument(
        "--max-eval-trajectories",
        type=int,
        default=6000,
        help="Cap for evaluation rollouts (large values need GPU memory).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--workers", type=int, default=None, help="Override hardware.workers.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile during training.")
    parser.add_argument(
        "--no-day10",
        action="store_true",
        help="Skip chronological Day 10 plots (no kaggle / faster).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job list and exit (no data gen, train, or eval).",
    )
    args = parser.parse_args()

    base_cfg = load_omega_config()
    if args.epochs is not None:
        base_cfg.training.epochs = int(args.epochs)
    if args.workers is not None:
        base_cfg.hardware.workers = int(args.workers)
    if args.no_compile:
        base_cfg.hardware.compile_model = False

    specs = {s.key: s for s in unique_data_specs()}
    jobs: list[dict[str, Any]] = [
        {"tag": "dt_mp_raw_K100", "data_key": "mp_raw_Hfull", "context_len": 100, "model_over": {}},
        {"tag": "dt_mp_shaped_K100", "data_key": "sh_raw_Hfull", "context_len": 100, "model_over": {}},
        {"tag": "dt_mp_logrt_K100", "data_key": "mp_logrt_Hfull", "context_len": 100, "model_over": {}},
        {"tag": "dt_sh_logrt_K100", "data_key": "sh_logrt_Hfull", "context_len": 100, "model_over": {}},
        {"tag": "dt_mp_raw_H50_K100", "data_key": "mp_raw_H50", "context_len": 100, "model_over": {}},
        {"tag": "dt_mp_raw_H100_K100", "data_key": "mp_raw_H100", "context_len": 100, "model_over": {}},
        {"tag": "dt_mp_raw_K50", "data_key": "mp_raw_Hfull", "context_len": 50, "model_over": {}},
        {"tag": "cnn_mp_raw_K100", "data_key": "mp_raw_Hfull", "context_len": 100, "model_over": {"architecture": "cnn"}},
        {
            "tag": "large_mp_raw_K100",
            "data_key": "mp_raw_Hfull",
            "context_len": 100,
            "model_over": {"d_model": 256, "n_layers": 4, "n_heads": 8},
        },
    ]

    if args.dry_run:
        print("DRY RUN — datasets (6000 train traj each by default):")
        for spec in specs.values():
            print(f"  data: {spec.key} | reward={spec.reward_type} state={spec.state_representation} H={spec.reward_horizon}")
        print("DRY RUN — train/eval jobs:")
        for j in jobs:
            print(f"  {j['tag']} | data={j['data_key']} K={j['context_len']} model={j['model_over']}")
        print("Then: context profile from dt_mp_raw_K50 + dt_mp_raw_K100 checkpoints.")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_root = (args.output_root / ts).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    ckpt_by_tag: dict[str, Path] = {}

    print(f"Run directory: {run_root}")

    # --- Phase 1: all datasets ---
    for spec in specs.values():
        ensure_dataset(
            spec,
            run_root,
            base_cfg,
            train_episodes=args.train_episodes,
            test_episodes=args.test_episodes,
        )

    # --- Phase 2: train + eval each job ---
    for j in jobs:
        spec = specs[j["data_key"]]
        train_path = run_root / "data" / f"{spec.key}_train.pt"
        test_path = run_root / "data" / f"{spec.key}_test.pt"
        job_dir = run_root / "jobs" / j["tag"]
        model_dir = job_dir / "model"
        plot_dir = job_dir / "plots"
        model_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg.training, resolve=True))
        train_cfg.context_len = int(j["context_len"])

        model_cfg = merge_model_cfg(base_cfg, j["model_over"])
        eval_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg.evaluation, resolve=True))
        eval_cfg.context_len = int(j["context_len"])
        eval_cfg.max_eval_trajectories = int(args.max_eval_trajectories)
        if args.no_day10:
            eval_cfg.continuous_day10_plot = False

        train_s, fitted_batch_size = train_with_batch_backoff(
            train_data_path=train_path,
            model_dir=model_dir,
            plot_dir=plot_dir,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            hardware_cfg=base_cfg.hardware,
        )

        reward_shaping_eval = OmegaConf.to_container(base_cfg.generator.reward_shaping, resolve=True)
        t1 = time.perf_counter()
        df_metrics = evaluate_model(
            model_path=str(model_dir / "dt_model_final.pt"),
            data_path=str(test_path),
            eval_cfg=eval_cfg,
            model_cfg=model_cfg,
            plot_dir=str(plot_dir),
            state_representation=spec.state_representation,
            train_data_path=str(train_path),
            generator_window_size=int(base_cfg.generator.window_size),
            generator_price_offset=float(base_cfg.generator.price_offset),
            generator_reward_type=str(spec.reward_type),
            generator_reward_shaping=reward_shaping_eval,
        )
        eval_s = time.perf_counter() - t1

        ckpt_by_tag[j["tag"]] = model_dir / "dt_model_final.pt"
        br = best_dt_row(df_metrics)
        row = {
            "tag": j["tag"],
            "data_key": spec.key,
            "reward_type": spec.reward_type,
            "state_representation": spec.state_representation,
            "reward_horizon": spec.reward_horizon,
            "context_len": j["context_len"],
            "architecture": OmegaConf.to_container(model_cfg).get("architecture", "transformer"),
            "d_model": int(model_cfg.d_model),
            "n_layers": int(model_cfg.n_layers),
            "batch_size": fitted_batch_size,
            "train_seconds": round(train_s, 2),
            "eval_seconds": round(eval_s, 2),
        }
        if len(br):
            row["DT_Sharpe"] = float(br.get("Sharpe", np.nan))
            row["DT_F1_macro"] = float(br.get("F1_macro", np.nan))
            row["DT_PnL"] = float(br.get("PnL", np.nan))
        aggregate_rows.append(row)
        manifest.append({**row, "model_dir": str(model_dir), "plots": str(plot_dir)})

        with open(run_root / "run_manifest.json", "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
        pd.DataFrame(aggregate_rows).to_csv(run_root / "aggregate_metrics.csv", index=False)

    # --- Phase 3: context horizon profile (transformer K=50 vs K=100, canonical data) ---
    k50_tag = "dt_mp_raw_K50"
    k100_tag = "dt_mp_raw_K100"
    if k50_tag in ckpt_by_tag and k100_tag in ckpt_by_tag:
        canon = specs["mp_raw_Hfull"]
        train_canon = run_root / "data" / f"{canon.key}_train.pt"
        test_canon = run_root / "data" / f"{canon.key}_test.pt"
        default_model = merge_model_cfg(base_cfg, {})
        model_dict = OmegaConf.to_container(default_model, resolve=True)
        prof_df = run_context_profile(
            run_root=run_root,
            ckpt_k50=ckpt_by_tag[k50_tag],
            ckpt_k100=ckpt_by_tag[k100_tag],
            train_data=train_canon,
            test_data=test_canon,
            model_cfg=model_dict,
            state_representation="raw",
            max_eval_trajectories=min(int(args.max_eval_trajectories), 512),
            rtg_rollout_mode=str(base_cfg.evaluation.rtg_rollout_mode),
        )
        print("\nContext profile (K=50 vs K=100):\n", prof_df.to_string(index=False))
        print(f"Context profile CSV/PNG: {run_root / 'context_profile'}")
    else:
        print("Skipping context profile: missing K50 or K100 checkpoints.")

    print(f"\nDone. Artifacts under {run_root}")


if __name__ == "__main__":
    main()
