#!/usr/bin/env python3
"""Statically quantize the ONNX model to int8 (QDQ) and compare fp32 vs int8 accuracy on the holdout split."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.metrics import roc_auc_score

from common import XrayDataset, eval_transform, read_manifest


class Reader(CalibrationDataReader):
    def __init__(self, rows, img_size, batch=8):
        self.ds = XrayDataset(rows, eval_transform(img_size))
        self.batch = batch
        self.i = 0

    def get_next(self):
        if self.i >= len(self.ds):
            return None
        xs = [self.ds[j][0] for j in range(self.i, min(self.i + self.batch, len(self.ds)))]
        self.i += self.batch
        return {"image": torch.stack(xs).numpy()}

    def rewind(self):
        self.i = 0


def run_onnx(path, rows, img_size, batch=16):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ds = XrayDataset(rows, eval_transform(img_size))
    out = []
    t0 = time.time()
    for i in range(0, len(ds), batch):
        x = torch.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))]).numpy()
        out.append(sess.run(None, {"image": x})[0])
    logits = np.concatenate(out)
    return 1 / (1 + np.exp(-logits)), (time.time() - t0) / len(ds)


def summarize(name, path, probs, labels, ref, secs):
    pred = probs >= 0.5
    m = dict(
        file=str(path), mb=round(Path(path).stat().st_size / 1e6, 1), auroc=float(roc_auc_score(labels, probs)),
        accuracy=float((pred == (labels == 1)).mean()), sensitivity=float(pred[labels == 1].mean()),
        specificity=float((~pred[labels == 0]).mean()), mean_abs_prob_diff_vs_torch=float(np.abs(probs - ref).mean()),
        max_abs_prob_diff_vs_torch=float(np.abs(probs - ref).max()), ms_per_image=1000 * secs,
    )
    print(f"{name:6} {m['mb']:5.1f} MB | AUROC {m['auroc']:.4f} | acc@0.5 {m['accuracy']:.4f} | sens {m['sensitivity']:.4f} | "
          f"spec {m['specificity']:.4f} | |dp| mean {m['mean_abs_prob_diff_vs_torch']:.4f} max {m['max_abs_prob_diff_vs_torch']:.4f} | "
          f"{m['ms_per_image']:.0f} ms/img", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--name", default="pneumonia_resnet18")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--predictions", default="reports/holdout_predictions.csv")
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-method", default="minmax", choices=["minmax", "percentile", "entropy"])
    ap.add_argument("--percentile", type=float, default=99.99)
    ap.add_argument("--reduce-range", action="store_true", help="7-bit weights; avoids u8s8 saturation on CPUs without VNNI")
    ap.add_argument("--suffix", default="int8")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    meta = json.loads((model_dir / f"{args.name}.json").read_text())
    img = meta["img_size"]
    fp32 = model_dir / f"{args.name}.onnx"
    pre = model_dir / f"{args.name}_preprocessed.onnx"
    int8 = model_dir / f"{args.name}_{args.suffix}.onnx"
    method = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
              "entropy": CalibrationMethod.Entropy}[args.calib_method]
    print(f"variant {args.suffix}: calib {args.calib_method} (percentile {args.percentile}), reduce_range {args.reduce_range}", flush=True)

    train_rows = read_manifest(args.manifest, "train")
    rng = np.random.default_rng(args.seed)
    calib_rows = [train_rows[i] for i in rng.choice(len(train_rows), args.calib, replace=False)]

    t0 = time.time()
    quant_pre_process(str(fp32), str(pre))
    quantize_static(str(pre), str(int8), Reader(calib_rows, img), quant_format=QuantFormat.QDQ, per_channel=True,
                    reduce_range=args.reduce_range, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    calibrate_method=method, extra_options={"CalibPercentile": args.percentile})
    pre.unlink()
    print(f"quantized with {len(calib_rows)} calibration images in {time.time() - t0:.0f}s", flush=True)

    holdout = read_manifest(args.manifest, "holdout")
    labels = np.array([int(r["label"]) for r in holdout])
    with open(args.predictions, newline="") as f:
        ref_rows = list(csv.DictReader(f))
    assert [r["path"] for r in ref_rows] == [r["path"] for r in holdout], "prediction order mismatch"
    ref = np.array([float(r["prob"]) for r in ref_rows])

    report = {}
    for name, path in (("fp32", fp32), ("int8", int8)):
        probs, secs = run_onnx(path, holdout, img)
        report[name] = summarize(name, path, probs, labels, ref, secs)
    report_path = model_dir / f"quantization_report_{args.suffix}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {int8} and {report_path}")


if __name__ == "__main__":
    main()
