#!/usr/bin/env python3
"""Export the trained classifier to ONNX and check it against PyTorch on real holdout images."""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from common import XrayDataset, eval_transform, load_checkpoint, read_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--name", default="pneumonia_resnet18")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--n-check", type=int, default=16)
    args = ap.parse_args()

    model, meta = load_checkpoint(args.model_dir, args.name)
    out = Path(args.model_dir) / f"{args.name}.onnx"
    dummy = torch.randn(1, 3, meta["img_size"], meta["img_size"])
    torch.onnx.export(model, dummy, str(out), input_names=["image"], output_names=["logit"],
                      dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}}, opset_version=17, dynamo=False)

    rows = read_manifest(args.manifest, "holdout")[: args.n_check]
    ds = XrayDataset(rows, eval_transform(meta["img_size"]))
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    with torch.no_grad():
        ref = torch.sigmoid(model(x)).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = 1 / (1 + np.exp(-sess.run(None, {"image": x.numpy()})[0]))
    diff = float(np.abs(ref - got).max())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB); max |torch - onnx| prob diff on {len(rows)} images: {diff:.2e}")
    assert diff < 1e-4, "ONNX output does not match PyTorch"


if __name__ == "__main__":
    main()
