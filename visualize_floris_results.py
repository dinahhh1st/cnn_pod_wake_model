"""
Visualize dual-branch POD-CNN predictions on a given dataset.

Produces per-sample figures (true wake, predicted wake, difference) and a summary
figure (R2 histogram, scatter, mean spatial error).

Usage
-----
Edit the path constants at the top of ``main()``, then run::

    python visualize_pod_cnn_opensource.py

Visual style: jet target/prediction, RdBu_r absolute error, per-panel min/max.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# -- project root discovery ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from Model.vector_dataset import VectorDataset
from Model.POD_CNN_v4 import DualBranchCNN_NoH400Dir_Ct


# ============================================================================
# Helper utilities (identical across field / FLORIS / custom datasets)
# ============================================================================

def resolve_split_path(dataset_dir: Path, split: str) -> Path:
    """Pick the first-existing NPZ for *split* from *dataset_dir*."""
    candidates = {
        "train": ("trainset.npz", "trainset_aug.npz"),
        "val": ("validationset.npz", "validationset_aug.npz"),
        "test": ("test.npz",),
    }
    for name in candidates[split]:
        path = dataset_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing {split} split under {dataset_dir}; tried {list(candidates[split])}"
    )


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


def sanitize_filename(value: str, max_len: int = 90) -> str:
    clean = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("._")
    return clean[:max_len] if clean else "sample"


# ============================================================================
# Inference & metrics
# ============================================================================

def collect_predictions(
    model: torch.nn.Module,
    dataset: VectorDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (prediction, target) in physical units (m/s)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pred_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for wp350, h400, target in loader:
            prediction = model(wp350.to(device), h400.to(device))
            pred_rows.append(prediction.cpu().numpy())
            target_rows.append(target.squeeze(1).cpu().numpy())
    pred_norm = np.concatenate(pred_rows, axis=0)
    target_norm = np.concatenate(target_rows, axis=0)
    pred_ms = pred_norm * dataset.wind_std + dataset.wind_mean
    target_ms = target_norm * dataset.wind_std + dataset.wind_mean
    return pred_ms.astype(np.float32), target_ms.astype(np.float32)


def per_sample_metrics(
    pred: np.ndarray, target: np.ndarray, filenames: np.ndarray
) -> pd.DataFrame:
    rows = []
    for idx in range(pred.shape[0]):
        y_pred = pred[idx].reshape(-1).astype(float)
        y_true = target[idx].reshape(-1).astype(float)
        error = y_pred - y_true
        sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
        sse = float(np.sum(error**2))
        rows.append(
            {
                "sample_index": idx,
                "source_filename": str(filenames[idx]),
                "r2": float(1.0 - sse / sst) if sst > 1e-12 else float("nan"),
                "mae_ms": float(np.mean(np.abs(error))),
                "rmse_ms": float(np.sqrt(np.mean(error**2))),
                "bias_ms": float(np.mean(error)),
                "true_mean_ms": float(np.mean(y_true)),
                "true_std_ms": float(np.std(y_true)),
                "pred_mean_ms": float(np.mean(y_pred)),
                "pred_std_ms": float(np.std(y_pred)),
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# Annotation helpers
# ============================================================================

def pod_annotation(
    row: pd.Series, n_modes: int, explained_energy: float | None
) -> str:
    lines = [
        f"R2 = {row['r2']:.4f}",
        f"MAE = {row['mae_ms']:.3f} m/s",
        f"RMSE = {row['rmse_ms']:.3f} m/s",
    ]
    return "\n".join(lines)


# ============================================================================
# Styling
# ============================================================================

def style_sample_axis(ax) -> None:
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.tick_params(
        axis="both", which="both", direction="in",
        labelsize=10, length=3.5, width=0.8,
    )


def style_sample_colorbar(
    cbar, label: str, vmin: float | None = None, vmax: float | None = None
) -> None:
    cbar.set_label(label, fontsize=12)
    if vmin is not None and vmax is not None:
        ticks = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(ticks)
        span = vmax - vmin
        if span < 1.0:
            cbar.set_ticklabels([f"{t:.3f}" for t in ticks])
        elif span < 10:
            cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
        else:
            cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
    cbar.ax.tick_params(
        which="both", direction="in", labelsize=10, length=3, width=0.8,
    )


# ============================================================================
# Core plotting
# ============================================================================

def plot_sample(
    output_path: Path,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    row: pd.Series,
    n_modes: int,
    explained_energy: float | None,
) -> None:
    """Save a 3-panel figure (true / predict / difference) for one sample."""
    abs_error = pred - true

    t_vmin, t_vmax = float(np.nanmin(true)), float(np.nanmax(true))
    p_vmin, p_vmax = float(np.nanmin(pred)), float(np.nanmax(pred))
    err_max = max(
        abs(float(np.nanpercentile(abs_error, 1))),
        abs(float(np.nanpercentile(abs_error, 99))),
        0.1,
    )
    extent = [
        float(grid_x[0]), float(grid_x[-1]),
        float(grid_y[0]), float(grid_y[-1]),
    ]

    cm = 1.0 / 2.54
    fig = plt.figure(figsize=(8.0 * cm, 16.0 * cm), dpi=900, constrained_layout=False)
    gs = fig.add_gridspec(
        4, 1, height_ratios=[1.0, 1.0, 1.0, 0.55], hspace=0.7,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    info_ax = fig.add_subplot(gs[3, 0])
    info_ax.axis("off")

    # -- panel (1) True -------------------------------------------------
    im0 = axes[0].imshow(
        true, origin="lower", extent=extent, aspect="auto",
        cmap="jet", vmin=t_vmin, vmax=t_vmax, interpolation="bilinear",
    )
    style_sample_axis(axes[0])
    axes[0].set_title("(1) True wake", loc="left", fontsize=12, fontweight="bold")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.035, pad=0.025)
    style_sample_colorbar(cbar0, "m/s", vmin=t_vmin, vmax=t_vmax)

    # -- panel (2) Predicted --------------------------------------------
    im1 = axes[1].imshow(
        pred, origin="lower", extent=extent, aspect="auto",
        cmap="jet", vmin=p_vmin, vmax=p_vmax, interpolation="bilinear",
    )
    style_sample_axis(axes[1])
    axes[1].set_title("(2) Predict wake", loc="left", fontsize=12, fontweight="bold")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.035, pad=0.025)
    style_sample_colorbar(cbar1, "m/s", vmin=p_vmin, vmax=p_vmax)

    # -- panel (3) Difference -------------------------------------------
    im2 = axes[2].imshow(
        abs_error, origin="lower", extent=extent, aspect="auto",
        cmap="RdBu_r", vmin=-err_max, vmax=err_max, interpolation="bilinear",
    )
    style_sample_axis(axes[2])
    axes[2].set_title("(3) Difference", loc="left", fontsize=12, fontweight="bold")
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.035, pad=0.025)
    style_sample_colorbar(cbar2, "m/s", vmin=-err_max, vmax=err_max)

    # -- metrics annotation ---------------------------------------------
    info_ax.text(
        0.02, 0.12,
        pod_annotation(row, n_modes, explained_energy),
        transform=info_ax.transAxes,
        fontsize=10, va="bottom", ha="left",
        linespacing=1.45,
    )

    fig.subplots_adjust(left=0.2, right=0.82, top=0.95, bottom=0.05)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600)
    plt.close(fig)


def plot_summary(
    output_path: Path,
    pred: np.ndarray,
    target: np.ndarray,
    metrics: pd.DataFrame,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> None:
    """Save a summary figure: R2 histogram, scatter, mean spatial error."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))

    axes[0].hist(
        metrics["r2"].dropna(), bins=40,
        color="#3b7a57", alpha=0.78, edgecolor="white",
    )
    axes[0].axvline(
        metrics["r2"].median(), color="#b24c35",
        linestyle="--", linewidth=1.2,
        label=f"median={metrics['r2'].median():.3f}",
    )
    axes[0].set_xlabel("Per-sample R2")
    axes[0].set_ylabel("count")
    axes[0].set_title("Test-sample R2 distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    gvmin = float(np.nanpercentile(target, 1))
    gvmax = float(np.nanpercentile(target, 99))
    axes[1].scatter(
        target.reshape(-1), pred.reshape(-1),
        s=0.18, alpha=0.12, color="#355c9a", edgecolors="none",
    )
    axes[1].plot(
        [gvmin, gvmax], [gvmin, gvmax],
        color="#b24c35", linestyle="--", linewidth=1,
    )
    axes[1].set_xlabel("True (m/s)")
    axes[1].set_ylabel("Predicted (m/s)")
    axes[1].set_title("All test pixels")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(alpha=0.25)

    mean_error = np.mean(pred - target, axis=0)
    emax = max(
        abs(float(np.nanmin(mean_error))),
        abs(float(np.nanmax(mean_error))),
        1e-6,
    )
    extent = [
        float(grid_x[0]), float(grid_x[-1]),
        float(grid_y[0]), float(grid_y[-1]),
    ]
    im = axes[2].imshow(
        mean_error, origin="lower", extent=extent, aspect="auto",
        cmap="RdBu_r", vmin=-emax, vmax=emax, interpolation="bilinear",
    )
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("y (m)")
    axes[2].set_title("Mean spatial error")
    plt.colorbar(im, ax=axes[2], label="m/s")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# POD energy
# ============================================================================

def compute_pod_energy(dataset: VectorDataset, n_modes: int) -> float | None:
    wind_grids = np.asarray(dataset.wind_grids, dtype=np.float32)
    if wind_grids.shape[0] < 2:
        return None
    mean_flow = np.mean(wind_grids, axis=0)
    fluctuation = (
        wind_grids.reshape(wind_grids.shape[0], -1) - mean_flow.reshape(1, -1)
    )
    singular_values = np.linalg.svd(
        fluctuation, full_matrices=False, compute_uv=False,
    )
    eigvals = singular_values ** 2
    total = float(np.sum(eigvals))
    if total <= 0.0:
        return None
    return float(np.sum(eigvals[:n_modes]) / total)


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    # ----- user-editable paths -----------------------------------------------
    DATASET_DIR = Path("./Dataset/floris_dataset")
    CHECKPOINT = Path("./checkpoints/best_model.pth")
    OUTPUT_DIR = Path("./outputs/figures")

    # ----- setup -------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    normalization = checkpoint["normalization"]
    n_modes = int(checkpoint.get("n_modes", 32))

    train_dataset = VectorDataset(
        resolve_split_path(DATASET_DIR, "train"), normalization=normalization,
    )
    test_dataset = VectorDataset(
        resolve_split_path(DATASET_DIR, "test"), normalization=normalization,
    )

    model = DualBranchCNN_NoH400Dir_Ct(
        h400_dim=8, wp350_dim=22, dropout_rate=0.3, n_modes=n_modes,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ----- inference ---------------------------------------------------------
    pred, target = collect_predictions(
        model, test_dataset, batch_size=32, device=device,
    )
    source_filenames = (
        np.asarray(
            np.load(
                resolve_split_path(DATASET_DIR, "test"),
                allow_pickle=True,
            )["source_filenames"]
        ).astype(str)
    )
    metrics = per_sample_metrics(pred, target, source_filenames)

    # ----- save raw predictions ----------------------------------------------
    raw_npz = OUTPUT_DIR / "test_predictions_raw_fields.npz"
    np.savez_compressed(
        raw_npz,
        prediction=pred.astype(np.float32),
        target=target.astype(np.float32),
    )
    metrics_path = OUTPUT_DIR / "per_sample_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    grid_x = np.asarray(test_dataset.grid_x, dtype=np.float32)
    grid_y = np.asarray(test_dataset.grid_y, dtype=np.float32)
    explained_energy = compute_pod_energy(train_dataset, n_modes)

    # ----- per-sample figures ------------------------------------------------
    sorted_metrics = metrics.sort_values("r2", na_position="first").reset_index(
        drop=True,
    )
    pick_positions = {
        "worst": 0,
        "q1": max(0, len(sorted_metrics) // 4),
        "median": max(0, len(sorted_metrics) // 2),
        "q3": max(0, len(sorted_metrics) * 3 // 4),
        "best": max(0, len(sorted_metrics) - 1),
    }
    selected_figures = []
    for label, position in pick_positions.items():
        row = sorted_metrics.iloc[position]
        idx = int(row["sample_index"])
        path = OUTPUT_DIR / f"pred_{label}_{idx}.png"
        plot_sample(
            path, grid_x, grid_y, target[idx], pred[idx],
            row, n_modes, explained_energy,
        )
        selected_figures.append(str(path))

    # ----- all-sample figures ------------------------------------------------
    all_dir = OUTPUT_DIR / "test_all_samples"
    all_dir.mkdir(parents=True, exist_ok=True)
    for _, row in metrics.sort_values("sample_index").iterrows():
        idx = int(row["sample_index"])
        r2_tag = f"{float(row['r2']):.3f}".replace("-", "m").replace(".", "p")
        name = sanitize_filename(row["source_filename"])
        path = all_dir / f"pred_sample_{idx:04d}_r2_{r2_tag}_{name}.png"
        plot_sample(
            path, grid_x, grid_y, target[idx], pred[idx],
            row, n_modes, explained_energy,
        )

    # ----- summary figure ----------------------------------------------------
    summary_fig = OUTPUT_DIR / "summary.png"
    plot_summary(summary_fig, pred, target, metrics, grid_x, grid_y)

    # ----- summary JSON ------------------------------------------------------
    summary = {
        "dataset_dir": str(DATASET_DIR),
        "checkpoint": str(CHECKPOINT),
        "output_dir": str(OUTPUT_DIR),
        "n_test_samples": int(len(metrics)),
        "n_modes": int(n_modes),
        "pod_energy_fraction": explained_energy,
        "global_mae_ms": float(np.mean(np.abs(pred - target))),
        "global_rmse_ms": float(np.sqrt(np.mean((pred - target) ** 2))),
        "per_sample_r2_mean": float(metrics["r2"].mean()),
        "per_sample_r2_median": float(metrics["r2"].median()),
        "per_sample_r2_min": float(metrics["r2"].min()),
        "per_sample_r2_max": float(metrics["r2"].max()),
        "raw_prediction_npz": str(raw_npz),
        "metrics_csv": str(metrics_path),
        "selected_figures": selected_figures,
        "all_sample_figures_dir": str(all_dir),
        "summary_figure": str(summary_fig),
        "visual_style": "jet target/prediction, RdBu_r absolute error, per-panel min/max.",
    }
    summary_path = OUTPUT_DIR / "visualization_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
