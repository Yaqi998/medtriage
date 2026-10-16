#!/usr/bin/env python3
"""Train the pneumonia classifier on the train split; keep the epoch with the best validation AUROC."""
import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from common import CLASS_NAMES, PneumoniaNet, XrayDataset, eval_transform, read_manifest, train_transform


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x)))
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def youden_threshold(probs, labels):
    fpr, tpr, thr = roc_curve(labels, probs)
    thr = np.clip(thr, 0.0, 1.0)
    return float(thr[int(np.argmax(tpr - fpr))])


def bce(probs, labels, eps=1e-7):
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--name", default="pneumonia_resnet18")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: train on only the first N images")
    args = ap.parse_args()

    set_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_rows = read_manifest(args.manifest, "train")
    val_rows = read_manifest(args.manifest, "val")
    if args.limit:
        train_rows = train_rows[: args.limit]
    counts = Counter(int(r["label"]) for r in train_rows)
    n_neg, n_pos = counts[0], counts[1]
    print(f"train {len(train_rows)} (normal {n_neg} / pneumonia {n_pos}) | val {len(val_rows)} | "
          f"torch threads {torch.get_num_threads()} | workers {args.workers}", flush=True)

    dl_kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    train_loader = DataLoader(XrayDataset(train_rows, train_transform(args.img)), batch_size=args.batch,
                              shuffle=True, **dl_kw)
    val_loader = DataLoader(XrayDataset(val_rows, eval_transform(args.img)), batch_size=64, shuffle=False, **dl_kw)

    model = PneumoniaNet(pretrained=True)
    head = model.trainable[-1]
    body = [p for n, p in model.trainable.named_parameters() if not n.startswith(str(len(model.trainable) - 1) + ".")]
    opt = torch.optim.AdamW([
        {"params": body, "lr": args.lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.wd)
    total_steps = args.epochs * len(train_loader)
    warmup = min(100, max(1, total_steps // 10))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos))

    best = {"val_auroc": -1.0}
    bad_epochs = 0
    log = []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, seen = 0.0, 0
        for b, (x, y) in enumerate(train_loader, 1):
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(y)
            seen += len(y)
            if b % 20 == 0 or b == len(train_loader):
                el = time.time() - t0
                print(f"epoch {epoch} [{b}/{len(train_loader)}] loss {run_loss / seen:.4f} | "
                      f"{seen / el:.1f} img/s | {el / 60:.1f} min", flush=True)
        train_loss = run_loss / seen

        vp, vl = predict(model, val_loader)
        val_auroc = float(roc_auc_score(vl, vp))
        val_acc = float(((vp >= 0.5) == (vl == 1)).mean())
        val_loss = bce(vp, vl)
        ep_min = (time.time() - t0) / 60
        print(f"== epoch {epoch}: train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"val_auroc {val_auroc:.4f} | val_acc@0.5 {val_acc:.4f} | {ep_min:.1f} min", flush=True)
        log.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_auroc=val_auroc,
                        val_acc=val_acc, minutes=ep_min))

        if val_auroc > best["val_auroc"] + 1e-4:
            best = dict(epoch=epoch, val_auroc=val_auroc, val_acc_at_0_5=val_acc, val_loss=val_loss,
                        threshold_youden=youden_threshold(vp, vl))
            torch.save(model.state_dict(), out / f"{args.name}.pt")
            with open(out / "val_predictions.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path", "patient", "label", "subtype", "prob"])
                for r, p in zip(val_rows, vp):
                    w.writerow([r["path"], r["patient"], r["label"], r["subtype"], f"{p:.6f}"])
            bad_epochs = 0
            print(f"   saved best (epoch {epoch}, val AUROC {val_auroc:.4f}, Youden t={best['threshold_youden']:.3f})",
                  flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop: no val AUROC improvement for {args.patience} epochs", flush=True)
                break

    with open(out / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)

    meta = dict(
        name=args.name, arch="resnet18 (ImageNet init; stem+layer1+layer2 frozen; layer3+layer4+head fine-tuned)",
        task="chest X-ray pneumonia screening (binary)", class_names=CLASS_NAMES, img_size=args.img,
        input="RGB-replicated grayscale, resize 256 -> center crop 224, ImageNet mean/std",
        dataset="Kermany et al. 2018, Mendeley Data rscbjbr9sj v2, CC BY 4.0",
        train_images=len(train_rows), train_normal=n_neg, train_pneumonia=n_pos, val_images=len(val_rows),
        best=best, threshold_default=0.5, args=vars(args), epochs_run=len(log),
        total_minutes=(time.time() - t_start) / 60, created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    (out / f"{args.name}.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {meta['total_minutes']:.1f} min; best epoch {best['epoch']} val AUROC {best['val_auroc']:.4f}; "
          f"wrote {out / (args.name + '.pt')} and .json", flush=True)


if __name__ == "__main__":
    main()
