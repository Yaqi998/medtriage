#!/usr/bin/env python3
"""Draw the ROC curve, confusion matrix and training curves for an evaluation report."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LIGHT_INK = "#f0efec"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "axes.titleweight": "semibold"})


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def plot_roc(labels, probs, metrics, split, out):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=160, facecolor=SURFACE)
    style(ax)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=(0, (4, 3)))
    ax.plot(fpr, tpr, color=BLUE, lw=2, solid_capstyle="round")
    points = []
    for key, label in (("default_0.5", "threshold 0.50"), ("val_tuned", None)):
        m = metrics["at_threshold"][key]
        points.append((1 - m["specificity"], m["sensitivity"], label or f"val-tuned threshold {m['threshold']:.2f}"))
    points.sort(key=lambda p: -p[1])  # higher point gets the upper label so leader lines never cross
    for (x, y, label), anchor in zip(points, [(0.42, 0.34), (0.42, 0.18)]):
        ax.plot(x, y, "o", ms=9, mfc=BLUE, mec=SURFACE, mew=2, zorder=5)
        ax.annotate(f"{label}\nsensitivity {100 * y:.1f}%  |  specificity {100 * (1 - x):.1f}%",
                    xy=(x, y), xytext=anchor, textcoords="data", fontsize=8.5, color=INK2, va="center",
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8, shrinkB=6))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"ROC curve - {split} set, AUROC {metrics['auroc']:.3f}", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(m, title, out):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    rates = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_aspect("equal")
    for i in range(2):
        for j in range(2):
            step = int(round(rates[i, j] * (len(SEQ) - 1)))
            ink = LIGHT_INK if step >= 7 else INK
            ink2 = LIGHT_INK if step >= 7 else INK2
            ax.add_patch(FancyBboxPatch((j + 0.02, i + 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.04",
                                        fc=SEQ[step], ec=SURFACE, lw=2))
            ax.text(j + 0.5, i + 0.45, f"{cm[i, j]:,}", ha="center", va="center", fontsize=17, fontweight="semibold", color=ink)
            ax.text(j + 0.5, i + 0.66, f"{100 * rates[i, j]:.1f}% of row", ha="center", va="center", fontsize=8.5, color=ink2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["predicted NORMAL", "predicted PNEUMONIA"])
    ax.xaxis.tick_top()
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["true\nNORMAL", "true\nPNEUMONIA"])
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=26)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_training(rows, best_epoch, out):
    ep = [int(r["epoch"]) for r in rows]
    tr = [float(r["train_loss"]) for r in rows]
    va = [float(r["val_loss"]) for r in rows]
    au = [float(r["val_auroc"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=160, facecolor=SURFACE)
    for ax in (a1, a2):
        style(ax)
        ax.set_xticks(ep)
        ax.set_xlabel("epoch")
    mk = dict(marker="o", ms=5.5, mec=SURFACE, mew=1.5, lw=2)
    a1.plot(ep, tr, color=BLUE, label="train loss", **mk)
    a1.plot(ep, va, color=ORANGE, label="val loss", **mk)
    for ys, name in ((tr, "train"), (va, "val")):
        a1.annotate(name, (ep[-1], ys[-1]), xytext=(7, 0), textcoords="offset points", fontsize=8.5, color=INK2, va="center")
    a1.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    a1.set_title("Loss (weighted BCE)", loc="left", fontsize=11)
    a1.margins(x=0.12)
    a2.plot(ep, au, color=BLUE, **mk)
    i = ep.index(best_epoch)
    a2.plot(ep[i], au[i], "o", ms=12, mfc="none", mec=INK2, mew=1.2)
    a2.annotate(f"best epoch {best_epoch}\nAUROC {au[i]:.4f}", (ep[i], au[i]), xytext=(10, -14), textcoords="offset points",
                fontsize=8.5, color=INK2, va="top")
    a2.set_title("Validation AUROC", loc="left", fontsize=11)
    a2.margins(x=0.12, y=0.25)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="reports/holdout_predictions.csv")
    ap.add_argument("--metrics", default="reports/holdout_metrics.json")
    ap.add_argument("--train-log", default="models/train_log.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--prefix", default="holdout")
    args = ap.parse_args()

    metrics = json.loads(Path(args.metrics).read_text())
    with open(args.predictions, newline="") as f:
        preds = list(csv.DictReader(f))
    labels = np.array([int(r["label"]) for r in preds])
    probs = np.array([float(r["prob"]) for r in preds])
    with open(args.train_log, newline="") as f:
        log_rows = list(csv.DictReader(f))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    split = metrics["split"]
    plot_roc(labels, probs, metrics, split, out / f"{args.prefix}_roc.png")
    m = metrics["at_threshold"]["val_tuned"]
    plot_confusion(m, f"Confusion matrix, {split} set, t = {m['threshold']:.2f}", out / f"{args.prefix}_confusion.png")
    plot_training(log_rows, int(metrics["best_epoch"]), out / "training_curves.png")
    print(f"wrote {out}/{args.prefix}_roc.png, {out}/{args.prefix}_confusion.png, {out}/training_curves.png")


if __name__ == "__main__":
    main()
