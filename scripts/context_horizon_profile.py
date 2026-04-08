#!/usr/bin/env python3
"""
Context horizon sweep: Sharpe ratio and macro-F1 vs instantaneous-return oracle
for trained Decision Transformers at different attention windows K.

Each K should be trained with matching ``training.context_len`` and evaluated with
the same K (checkpoints are not interchangeable across K without retraining).

DeepLOB (Zhang et al., 2018) reports FI-2010 *mid-price movement* classification
F1 at prediction horizons k in {10, 50, 100} — a different dataset and label
construction than our oracle-directional F1; reference numbers are printed and
plotted for qualitative context only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Repo root on path when run as ``python scripts/context_horizon_profile.py``
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from omegaconf import OmegaConf

from src.evaluations.direction_metrics import compute_directional_f1
from src.evaluations.dt_viz import compute_financial_metrics, vectorized_autoregressive_rollout
from src.evaluations.market_returns import get_market_returns
from src.models.decision_transformer import DecisionTransformer

# Table I (Setup 1), DeepLOB macro-style reporting — F1 from paper summary (%).
DEELOB_FI2010_SETUP1_F1 = {
    10: 77.66,
    50: 74.96,
    100: 76.58,
}


def _parse_kv_pairs(pairs: list[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Expected K=path, got: {p!r}")
        k_s, path_s = p.split("=", 1)
        out[int(k_s.strip())] = Path(path_s.strip()).expanduser().resolve()
    return dict(sorted(out.items()))


def _load_model(model_path: Path, model_cfg, device: torch.device) -> DecisionTransformer:
    model = DecisionTransformer(
        state_dim=model_cfg.state_dim,
        act_dim=model_cfg.act_dim,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        max_timestep=model_cfg.max_timestep,
        dropout=model_cfg.dropout,
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    raw = checkpoint["model_state_dict"]
    clean = {k.replace("_orig_mod.", ""): v for k, v in raw.items()}
    model.load_state_dict(clean)
    model.eval()
    return model


def run_profile(
    checkpoints: dict[int, Path],
    test_data: Path,
    plot_dir: Path,
    target_rtg: float,
    max_eval_trajectories: int,
    state_representation: str,
    model_cfg: dict,
    rtg_rollout_mode: str = "anchored_offline",
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc = OmegaConf.create(model_cfg)

    trajectories = torch.load(test_data, map_location="cpu", weights_only=False)
    if isinstance(trajectories, dict):
        trajectories = list(trajectories.values())
    trajectories = trajectories[:max_eval_trajectories]
    min_len = min(len(traj["states"]) for traj in trajectories)
    states_batch = torch.stack(
        [torch.tensor(traj["states"][:min_len], dtype=torch.float32) for traj in trajectories]
    )[:, :, : mc.state_dim].to(device)

    market_returns = get_market_returns(states_batch, state_representation=state_representation)

    reference_rtg = torch.stack(
        [
            torch.tensor(traj["rtg"][:min_len, 0], dtype=torch.float32)
            for traj in trajectories
        ]
    ).to(device)

    rows = []
    for K, ckpt in checkpoints.items():
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing checkpoint for K={K}: {ckpt}")
        model = _load_model(ckpt, mc, device)
        rr_kw = dict(
            model=model,
            states=states_batch,
            market_returns=market_returns,
            target_rtg=target_rtg,
            context_len=K,
            device=device,
            max_timestep=int(getattr(mc, "max_timestep", 10_000)),
            rtg_rollout_mode=rtg_rollout_mode,
        )
        if rtg_rollout_mode == "anchored_offline":
            rr_kw["reference_rtg"] = reference_rtg
        rewards, _, actions = vectorized_autoregressive_rollout(**rr_kw)
        fin = compute_financial_metrics(rewards)
        f1 = compute_directional_f1(actions, market_returns)
        rows.append(
            {
                "context_K": K,
                "Sharpe": fin["Sharpe"],
                "F1_macro": f1,
                "PnL": fin["PnL"],
                "MaxDD": fin["MaxDD"],
                "checkpoint": str(ckpt),
            }
        )

    df = pd.DataFrame(rows).sort_values("context_K").reset_index(drop=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_path = plot_dir / "context_horizon_profile.csv"
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["context_K"], df["Sharpe"], "o-", color="tab:blue")
    axes[0].set_xlabel("Context length K")
    axes[0].set_ylabel("Sharpe (unannualized)")
    axes[0].set_title("DT rollout vs K")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["context_K"], df["F1_macro"] * 100.0, "s-", color="tab:green")
    axes[1].set_xlabel("Context length K (DT attention window)")
    axes[1].set_ylabel("Macro-F1 vs oracle (%)")
    axes[1].set_title("Directional agreement (instantaneous mid-proxy)")
    axes[1].grid(True, alpha=0.3)
    deeplob_txt = (
        "DeepLOB FI-2010 Table I (Setup 1, movement F1 %):\n"
        + ", ".join(f"k={k}: {v:.2f}" for k, v in sorted(DEELOB_FI2010_SETUP1_F1.items()))
        + "\n(k = prediction steps; not our context K)"
    )
    axes[1].text(
        0.02,
        0.02,
        deeplob_txt,
        transform=axes[1].transAxes,
        fontsize=7,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )

    fig.suptitle(
        f"Context horizon profile (RTG={target_rtg}, {state_representation}, rollout={rtg_rollout_mode})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "context_horizon_profile.png", dpi=200)
    plt.close()

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile DT Sharpe and F1 across context lengths K.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Pairs K=/path/to/dt_model_final.pt e.g. 50=models/k50/dt_model_final.pt",
    )
    parser.add_argument("--test_data", type=Path, default=_ROOT / "data" / "test_trajectories.pt")
    parser.add_argument("--plot_dir", type=Path, default=_ROOT / "plots")
    parser.add_argument("--target_rtg", type=float, default=1.0)
    parser.add_argument("--max_eval_trajectories", type=int, default=32)
    parser.add_argument(
        "--state_representation",
        choices=("raw", "log_returns", "bps"),
        default="raw",
    )
    parser.add_argument("--state_dim", type=int, default=41)
    parser.add_argument("--act_dim", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--max_timestep", type=int, default=10000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--rtg_rollout_mode",
        choices=("autoregressive", "anchored_offline"),
        default="anchored_offline",
        help="RTG conditioning during rollout (see src/evaluations/dt_viz.py).",
    )
    args = parser.parse_args()

    ckpt_map = _parse_kv_pairs(args.checkpoints)
    model_cfg = {
        "state_dim": args.state_dim,
        "act_dim": args.act_dim,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "max_timestep": args.max_timestep,
        "dropout": args.dropout,
    }

    df = run_profile(
        checkpoints=ckpt_map,
        test_data=args.test_data.resolve(),
        plot_dir=args.plot_dir.resolve(),
        target_rtg=args.target_rtg,
        max_eval_trajectories=args.max_eval_trajectories,
        state_representation=args.state_representation,
        model_cfg=model_cfg,
        rtg_rollout_mode=args.rtg_rollout_mode,
    )

    print(df.to_string(index=False))
    print(f"\nWrote {args.plot_dir / 'context_horizon_profile.csv'}")
    print(f"Wrote {args.plot_dir / 'context_horizon_profile.png'}")
    print(
        "\nDeepLOB reference (FI-2010, Setup 1, F1 %): "
        + ", ".join(f"k={k}: {v:.2f}%" for k, v in sorted(DEELOB_FI2010_SETUP1_F1.items()))
    )


if __name__ == "__main__":
    main()
