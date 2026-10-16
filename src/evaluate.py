#!/usr/bin/env python3
"""Evaluate the trained classifier once on a split (default: holdout) and write metrics, predictions and a report."""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from common import XrayDataset, eval_transform, load_checkpoint, read_manifest


def metrics_at(probs, labels, t):
    pred = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()

    def div(a, b):
        return float(a / b) if b else float("nan")

    return dict(threshold=float(t), tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                accuracy=div(tp + tn, tp + tn + fp + fn), sensitivity=div(tp, tp + fn),
                specificity=div(tn, tn + fp), ppv=div(tp, tp + fp), npv=div(tn, tn + fn),
                f1=div(2 * tp, 2 * tp + fp + fn))


def bootstrap_ci(probs, labels, t, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(labels)
    samples = defaultdict(list)
    for _ in range(n):
        idx = rng.integers(0, N, N)
        p, l = probs[idx], labels[idx]
        if l.min() == l.max():
            continue
        samples["auroc"].append(roc_auc_score(l, p))
        m = metrics_at(p, l, t)
        for k in ("accuracy", "sensitivity", "specificity"):
            samples[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()}


@torch.no_grad()
def predict(model, loader):
    probs = []
    for x, _ in loader:
        probs.append(torch.sigmoid(model(x)))
    return torch.cat(probs).numpy()


def pct(x):
    return "n/a" if x != x else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--name", default="pneumonia_resnet18")
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    model, meta = load_checkpoint(args.model_dir, args.name)
    rows = read_manifest(args.manifest, args.split)
    labels = np.array([int(r["label"]) for r in rows])
    loader = DataLoader(XrayDataset(rows, eval_transform(meta["img_size"])), batch_size=args.batch,
                        shuffle=False, num_workers=args.workers)
    t0 = time.time()
    probs = predict(model, loader)
    infer_s = time.time() - t0

    t_tuned = float(meta["best"]["threshold_youden"])
    thresholds = {"default_0.5": 0.5, "val_tuned": t_tuned}
    result = dict(
        split=args.split, n_images=len(rows), n_patients=len({r["patient"] for r in rows}),
        n_normal=int((labels == 0).sum()), n_pneumonia=int((labels == 1).sum()),
        model=meta["arch"], best_epoch=meta["best"]["epoch"], val_auroc=meta["best"]["val_auroc"],
        auroc=float(roc_auc_score(labels, probs)), auprc=float(average_precision_score(labels, probs)),
        at_threshold={k: metrics_at(probs, labels, t) for k, t in thresholds.items()},
        ci95_val_tuned=bootstrap_ci(probs, labels, t_tuned),
        inference_ms_per_image=1000 * infer_s / len(rows),
    )

    by_sub = defaultdict(list)
    for r, p in zip(rows, probs):
        if r["label"] == "1":
            by_sub[r["subtype"]].append(p >= t_tuned)
    result["sensitivity_by_subtype_val_tuned"] = {k: dict(n=len(v), sensitivity=float(np.mean(v))) for k, v in by_sub.items()}

    by_orig = defaultdict(lambda: [[], []])
    for r, p in zip(rows, probs):
        by_orig[r["orig_split"]][0].append(int(r["label"]))
        by_orig[r["orig_split"]][1].append(p)
    result["by_original_dataset_split"] = {
        k: dict(n=len(l), auroc=float(roc_auc_score(l, p)) if min(l) != max(l) else None,
                accuracy_val_tuned=metrics_at(np.array(p), np.array(l), t_tuned)["accuracy"])
        for k, (l, p) in by_orig.items()
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.split}_metrics.json").write_text(json.dumps(result, indent=2))
    with open(out / f"{args.split}_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "patient", "label", "subtype", "orig_split", "prob"])
        for r, p in zip(rows, probs):
            w.writerow([r["path"], r["patient"], r["label"], r["subtype"], r["orig_split"], f"{p:.6f}"])

    d, v = result["at_threshold"]["default_0.5"], result["at_threshold"]["val_tuned"]
    ci = result["ci95_val_tuned"]
    lines = [
        f"# Pneumonia screening - {args.split} evaluation",
        "",
        f"Model: {meta['arch']}, {meta['img_size']}px input, best epoch {meta['best']['epoch']} "
        f"(val AUROC {meta['best']['val_auroc']:.4f}).",
        f"{args.split.capitalize()} set: {result['n_images']} images from {result['n_patients']} patients "
        f"({result['n_normal']} normal / {result['n_pneumonia']} pneumonia), patient-disjoint from the other splits.",
        "",
        f"AUROC {result['auroc']:.4f} (95% CI {ci['auroc'][0]:.3f}-{ci['auroc'][1]:.3f}) | AUPRC {result['auprc']:.4f}",
        "",
        "| Metric | threshold 0.5 | val-tuned threshold {:.3f} | 95% CI (tuned) |".format(t_tuned),
        "|---|---|---|---|",
        f"| Accuracy | {pct(d['accuracy'])} | {pct(v['accuracy'])} | {pct(ci['accuracy'][0])} - {pct(ci['accuracy'][1])} |",
        f"| Sensitivity (recall) | {pct(d['sensitivity'])} | {pct(v['sensitivity'])} | {pct(ci['sensitivity'][0])} - {pct(ci['sensitivity'][1])} |",
        f"| Specificity | {pct(d['specificity'])} | {pct(v['specificity'])} | {pct(ci['specificity'][0])} - {pct(ci['specificity'][1])} |",
        f"| PPV (precision) | {pct(d['ppv'])} | {pct(v['ppv'])} | |",
        f"| NPV | {pct(d['npv'])} | {pct(v['npv'])} | |",
        f"| F1 | {pct(d['f1'])} | {pct(v['f1'])} | |",
        "",
        f"Confusion matrix at tuned threshold (rows = truth, cols = predicted):",
        "",
        "| | pred NORMAL | pred PNEUMONIA |",
        "|---|---|---|",
        f"| true NORMAL | {v['tn']} | {v['fp']} |",
        f"| true PNEUMONIA | {v['fn']} | {v['tp']} |",
        "",
        "Sensitivity by pneumonia subtype (tuned threshold): "
        + ", ".join(f"{k} {pct(s['sensitivity'])} (n={s['n']})" for k, s in result["sensitivity_by_subtype_val_tuned"].items()),
        "",
        "By original dataset folder: "
        + ", ".join(f"{k}: n={s['n']}, AUROC {s['auroc']:.4f}, acc {pct(s['accuracy_val_tuned'])}"
                    for k, s in result["by_original_dataset_split"].items()),
        "",
        f"CPU inference: {result['inference_ms_per_image']:.0f} ms/image (batched).",
    ]
    (out / f"{args.split}_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
