#!/usr/bin/env python3
"""Build the browser model: fold BatchNorm into convolutions, store large weights as float16 (cast back to
float32 inside the graph, so compute stays fp32), and verify holdout accuracy against the PyTorch predictions."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from common import read_manifest
from quantize_onnx import run_onnx, summarize


def optimize_basic(src, dst):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])


def to_fp16_storage(src, dst, min_elems=1024):
    m = onnx.load(str(src))
    g = m.graph
    casts, converted, kept = [], 0, 0
    for init in list(g.initializer):
        arr = numpy_helper.to_array(init) if init.data_type == TensorProto.FLOAT else None
        if arr is None or arr.size < min_elems or np.abs(arr).max() > 65000:
            kept += 1
            continue
        g.initializer.remove(init)
        g.initializer.append(numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16"))
        casts.append(helper.make_node("Cast", [init.name + "_fp16"], [init.name], to=TensorProto.FLOAT, name="cast_" + init.name))
        converted += 1
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    return converted, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--name", default="pneumonia_resnet18")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--predictions", default="reports/holdout_predictions.csv")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    meta = json.loads((model_dir / f"{args.name}.json").read_text())
    fp32 = model_dir / f"{args.name}.onnx"
    folded = model_dir / f"{args.name}_folded.onnx"
    web = model_dir / f"{args.name}_fp16w.onnx"

    optimize_basic(fp32, folded)
    ops = Counter((n.domain or "ai.onnx", n.op_type) for n in onnx.load(str(folded)).graph.node)
    print("ops after BatchNorm folding:", ", ".join(f"{op}x{c}" for (dom, op), c in sorted(ops.items())))
    assert all(dom == "ai.onnx" for dom, _ in ops), "non-standard ops would not be portable"
    converted, kept = to_fp16_storage(folded, web)
    folded.unlink()
    print(f"stored {converted} weight tensors as float16 ({kept} small ones kept float32)")

    holdout = read_manifest(args.manifest, "holdout")
    labels = np.array([int(r["label"]) for r in holdout])
    with open(args.predictions, newline="") as f:
        ref = np.array([float(r["prob"]) for r in csv.DictReader(f)])
    report = {}
    for name, path in (("fp32", fp32), ("fp16w", web)):
        probs, secs = run_onnx(path, holdout, meta["img_size"])
        report[name] = summarize(name, path, probs, labels, ref, secs)
    (model_dir / "web_model_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {web}")


if __name__ == "__main__":
    main()
