#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plot_ftt_training.py
Parse FT-Transformer training logs and produce publication-ready plots.

Usage examples:
  python plot_ftt_training.py --log-file ftt_b1.log --out-dir figs_b1
  # 加上预测CSV可以画误差分布：
  python plot_ftt_training.py --log-file ftt_b1.log --out-dir figs_b1 \
      --pred-csv out_b1_reg/test_pred_ftt.csv
"""

import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPOCH_LINE = re.compile(
    r"\[Epoch\s+(\d{3})\]\s+tr_loss=([0-9.]+)\s+va_loss=([0-9.]+)\s+\|\s+rmse=([0-9.]+)"
)
TEST_METRICS = re.compile(r"\[FTT\]\s+Test RMSE=([0-9.]+)\s+MAE=([0-9.]+)")

def parse_log(text: str) -> pd.DataFrame:
    epochs, tr_loss, va_loss, rmse = [], [], [], []
    for m in EPOCH_LINE.finditer(text):
        epochs.append(int(m.group(1)))
        tr_loss.append(float(m.group(2)))
        va_loss.append(float(m.group(3)))
        rmse.append(float(m.group(4)))
    if not epochs:
        raise ValueError("No epoch lines matched. Check the log format.")
    df = pd.DataFrame({
        "epoch": epochs,
        "train_loss": tr_loss,
        "val_loss": va_loss,
        "val_rmse": rmse
    })
    return df

def parse_test_metrics(text: str):
    m = TEST_METRICS.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None

def smooth_series(s: pd.Series, win: int = 5) -> pd.Series:
    return s.rolling(window=win, min_periods=1, center=True).mean()

def ensure_out(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

def plot_loss_curves(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8,5), dpi=160)
    plt.plot(df["epoch"], smooth_series(df["train_loss"]), label="Train loss (smooth)")
    plt.plot(df["epoch"], smooth_series(df["val_loss"]), label="Val loss (smooth)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FT-Transformer — Train vs. Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_val_rmse(df: pd.DataFrame, out_path: Path):
    best_idx = int(df["val_rmse"].idxmin())
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_val_rmse = float(df.loc[best_idx, "val_rmse"])

    plt.figure(figsize=(8,5), dpi=160)
    plt.plot(df["epoch"], df["val_rmse"], label="Val RMSE (epoch)")
    plt.plot(df["epoch"], smooth_series(df["val_rmse"]), label="Val RMSE (smooth)")
    plt.scatter([best_epoch], [best_val_rmse], s=40,
                label=f"Best @ {best_epoch} (RMSE={best_val_rmse:.4f})")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (pixel domain)")
    plt.title("FT-Transformer — Validation RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return best_epoch, best_val_rmse

def plot_test_bars(best_val_rmse: float, test_rmse: float, test_mae: float, out_path: Path):
    plt.figure(figsize=(6,5), dpi=160)
    labels = ["Best Val RMSE", "Test RMSE", "Test MAE"]
    vals = [best_val_rmse, test_rmse, test_mae]
    plt.bar(labels, vals)
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.ylabel("Pixels")
    plt.title("FT-Transformer — Test Metrics")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_error_distribution(pred_csv: Path, out_hist: Path, out_cdf: Path):
    """
    支持两种列名：
      - 归一化域：err_norm
      - 反归一化（像素/米）：err_denorm（优先）
    """
    dfp = pd.read_csv(pred_csv)
    if "err_denorm" in dfp.columns:
        err = dfp["err_denorm"].values
        unit = "pixels (de-normalized)"
    elif "err_norm" in dfp.columns:
        err = dfp["err_norm"].values
        unit = "label domain"
    else:
        # 若没有误差列，尝试用 y_true / y_pred 计算
        cols_needed = {"y_true_0", "y_true_1", "y_pred_0", "y_pred_1"}
        if cols_needed.issubset(set(dfp.columns)):
            dx = dfp["y_pred_0"].values - dfp["y_true_0"].values
            dy = dfp["y_pred_1"].values - dfp["y_true_1"].values
            err = np.sqrt(dx*dx + dy*dy)
            unit = "label domain"
        else:
            print("Skip error plots: no error columns found.")
            return

    # Histogram
    plt.figure(figsize=(7,5), dpi=160)
    plt.hist(err, bins=40)
    plt.xlabel(f"Error norm [{unit}]")
    plt.ylabel("Count")
    plt.title("Test Error Distribution (Histogram)")
    plt.tight_layout()
    plt.savefig(out_hist, bbox_inches="tight")
    plt.close()

    # Empirical CDF
    xs = np.sort(err)
    ys = np.arange(1, len(xs)+1) / len(xs)
    plt.figure(figsize=(7,5), dpi=160)
    plt.plot(xs, ys)
    plt.xlabel(f"Error norm [{unit}]")
    plt.ylabel("Empirical CDF")
    plt.title("Test Error CDF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_cdf, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-file", type=Path, required=True, help="Training log file to parse.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to save figures/CSVs.")
    ap.add_argument("--pred-csv", type=Path, default=None,
                    help="Optional: test_pred_ftt.csv for error distribution plots.")
    args = ap.parse_args()

    ensure_out(args.out_dir)
    text = args.log_file.read_text(encoding="utf-8", errors="ignore")

    # Parse epochs
    df = parse_log(text)
    # Save CSV
    csv_path = args.out_dir / "ftt_training_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Plots
    loss_fig = args.out_dir / "ftt_loss_curve.png"
    rmse_fig = args.out_dir / "ftt_val_rmse.png"
    plot_loss_curves(df, loss_fig)
    best_epoch, best_val_rmse = plot_val_rmse(df, rmse_fig)

    # Test metrics (if present in log)
    test_rmse, test_mae = parse_test_metrics(text)
    if (test_rmse is not None) and (test_mae is not None):
        bar_fig = args.out_dir / "ftt_test_metrics.png"
        plot_test_bars(best_val_rmse, test_rmse, test_mae, bar_fig)
        print(f"[INFO] Test metrics found in log: RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")
    else:
        print("[INFO] No [FTT] test metrics line found; skip test bar plot.")

    # Optional error plots from prediction CSV
    if args.pred_csv is not None and args.pred_csv.exists():
        hist_fig = args.out_dir / "ftt_error_hist.png"
        cdf_fig  = args.out_dir / "ftt_error_cdf.png"
        plot_error_distribution(args.pred_csv, hist_fig, cdf_fig)
        print(f"[OK] Error plots saved: {hist_fig.name}, {cdf_fig.name}")

    print(f"[OK] Best val RMSE at epoch {best_epoch}: {best_val_rmse:.4f}")
    print(f"[OK] Saved curves to: {args.out_dir.resolve()}")
    print(f"[OK] Epoch CSV: {csv_path.name}")

if __name__ == "__main__":
    main()

    # 把txt中的实验结果变成csv
#
# python plot.py --log-file transformer_output_site1b1.txt --out-dir figs_b1
#
