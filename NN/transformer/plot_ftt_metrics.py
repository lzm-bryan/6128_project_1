#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_ftt_metrics.py — visualize FT-Transformer training log (loss / RMSE / MAE)
Usage:
  python plot_ftt_metrics.py --log transformer_output_site1b1.txt --out figs_b1
"""

import os, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_log(log_path):
    epochs, tr_loss, va_loss, rmse = [], [], [], []
    test_rmse, test_mae = None, None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Epoch lines
            m = re.match(r"\[Epoch\s+(\d+)\].*tr_loss=([\d\.]+).*va_loss=([\d\.]+).*rmse=([\d\.]+)", line)
            if m:
                epochs.append(int(m.group(1)))
                tr_loss.append(float(m.group(2)))
                va_loss.append(float(m.group(3)))
                rmse.append(float(m.group(4)))
            # Final test metrics
            if "Test RMSE=" in line:
                try:
                    test_rmse = float(re.search(r"Test RMSE=([\d\.]+)", line).group(1))
                    test_mae  = float(re.search(r"MAE=([\d\.]+)", line).group(1))
                except:
                    pass

    df = pd.DataFrame({
        "epoch": epochs,
        "train_loss": tr_loss,
        "val_loss": va_loss,
        "val_rmse": rmse
    })
    return df, test_rmse, test_mae

def plot_curves(df, out_dir, test_rmse=None, test_mae=None):
    os.makedirs(out_dir, exist_ok=True)

    # ---- (1) Train vs Val Loss ----
    plt.figure(figsize=(7,5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", lw=2)
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", lw=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend(); plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ftt_loss_curve.png"), dpi=300)
    plt.close()

    # ---- (2) Validation RMSE ----
    plt.figure(figsize=(7,5))
    plt.plot(df["epoch"], df["val_rmse"], color="tab:orange", lw=2)
    best_idx = np.argmin(df["val_rmse"])
    plt.axvline(df["epoch"].iloc[best_idx], color="gray", ls="--", label=f"Best (epoch {df['epoch'].iloc[best_idx]})")
    plt.scatter(df["epoch"].iloc[best_idx], df["val_rmse"].iloc[best_idx], color="red", zorder=5)
    plt.xlabel("Epoch"); plt.ylabel("Validation RMSE")
    plt.title("Validation RMSE per Epoch")
    plt.legend(); plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ftt_val_rmse.png"), dpi=300)
    plt.close()

    # ---- (3) Test Metrics ----
    if test_rmse is not None:
        plt.figure(figsize=(5,4))
        bars = ["Best Val RMSE", "Test RMSE", "Test MAE"]
        vals = [df["val_rmse"].min(), test_rmse, test_mae]
        plt.bar(bars, vals, color=["#66c2a5","#fc8d62","#8da0cb"])
        for i, v in enumerate(vals):
            plt.text(i, v + 0.002, f"{v:.3f}", ha='center', va='bottom', fontsize=10)
        plt.ylabel("Error")
        plt.title("Model Performance Summary")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ftt_test_metrics.png"), dpi=300)
        plt.close()

    print(f"✅ Saved all plots to: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="training log file")
    ap.add_argument("--out", required=True, help="output directory for plots")
    args = ap.parse_args()

    df, test_rmse, test_mae = parse_log(args.log)
    print(df.head())
    print(f"Test RMSE={test_rmse}, MAE={test_mae}")
    plot_curves(df, args.out, test_rmse, test_mae)

# 把csv中的结果画图
# python plot_ftt_metrics.py --log transformer_output_site1b1.txt --out figs_b1
