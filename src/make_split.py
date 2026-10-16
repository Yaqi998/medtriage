#!/usr/bin/env python3
"""Inventory the Kermany chest X-ray images, drop exact duplicates, group images by patient,
and write a stratified, patient-disjoint train / val / holdout split."""
import argparse
import csv
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PNEU_RE = re.compile(r"^person(\d+)_(bacteria|virus)_(\d+)", re.I)
NORM_RE = re.compile(r"^(NORMAL2-)?IM-(\d+)-(\d+)", re.I)
LABELS = {"NORMAL": 0, "PNEUMONIA": 1}


def patient_key(stem: str, label_name: str, orig_split: str):
    """Best-effort patient id from the filename. Train/test numbering are separate namespaces."""
    if label_name == "PNEUMONIA":
        m = PNEU_RE.match(stem)
        if m:
            return f"{orig_split}/person{m.group(1)}", m.group(2).lower()
    else:
        m = NORM_RE.match(stem)
        if m:
            series = "normal2" if m.group(1) else "normal"
            return f"{orig_split}/{series}-{m.group(2)}", "normal"
    return f"{orig_split}/{stem}", "unknown"


def md5sum(path: Path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assign_splits(groups_by_label, fracs, seed):
    rng = random.Random(seed)
    split_of = {}
    for label_name, groups in sorted(groups_by_label.items()):
        gids = sorted(groups)
        rng.shuffle(gids)
        total = sum(len(groups[g]) for g in gids)
        cut_train = fracs[0] * total
        cut_val = (fracs[0] + fracs[1]) * total
        seen = 0
        for g in gids:
            mid = seen + len(groups[g]) / 2
            split_of[g] = "train" if mid < cut_train else "val" if mid < cut_val else "holdout"
            seen += len(groups[g])
    return split_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="data/raw")
    ap.add_argument("--out", default="data/splits.csv")
    ap.add_argument("--fracs", default="0.70,0.15,0.15")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    fracs = [float(x) for x in args.fracs.split(",")]
    assert abs(sum(fracs) - 1) < 1e-6 and len(fracs) == 3

    raw_root = Path(args.raw_root)
    files = sorted(p for p in raw_root.rglob("*")
                   if p.suffix.lower() in {".jpeg", ".jpg", ".png"} and not p.name.startswith(".")
                   and "__MACOSX" not in p.parts)
    if not files:
        sys.exit(f"no images under {raw_root}")

    records, seen_md5, dupes = [], {}, 0
    for p in files:
        label_name = p.parent.name.upper()
        if label_name not in LABELS:
            continue
        digest = md5sum(p)
        if digest in seen_md5:
            dupes += 1
            continue
        seen_md5[digest] = p
        orig_split = p.parent.parent.name
        patient, subtype = patient_key(p.stem, label_name, orig_split)
        records.append(dict(
            path=str(p.relative_to(raw_root)), label=LABELS[label_name], label_name=label_name,
            subtype=subtype, patient=patient, orig_split=orig_split, md5=digest,
        ))

    groups_by_label = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups_by_label[r["label_name"]][r["patient"]].append(r)
    split_of = assign_splits(groups_by_label, fracs, args.seed)
    for r in records:
        r["split"] = split_of[r["patient"]]

    patients_per_split = defaultdict(set)
    for r in records:
        patients_per_split[r["split"]].add(r["patient"])
    for a in patients_per_split:
        for b in patients_per_split:
            if a < b:
                assert not (patients_per_split[a] & patients_per_split[b]), f"patient overlap {a}/{b}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "label", "label_name", "subtype", "patient", "orig_split", "split", "md5"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    unknown = sum(r["subtype"] == "unknown" for r in records)
    print(f"images found: {len(files)} | exact duplicates dropped: {dupes} | kept: {len(records)} | unparsed names: {unknown}")
    print(f"patients (groups): {len(split_of)}")
    print(f"{'split':8} {'images':>7} {'patients':>9} {'normal':>7} {'pneum':>7} {'bact':>6} {'virus':>6} {'origTrain':>10} {'origTest':>9}")
    for s in ["train", "val", "holdout"]:
        rs = [r for r in records if r["split"] == s]
        c = Counter(r["subtype"] for r in rs)
        o = Counter(r["orig_split"] for r in rs)
        print(f"{s:8} {len(rs):7d} {len(patients_per_split[s]):9d} {c['normal']:7d} {c['bacteria'] + c['virus']:7d} "
              f"{c['bacteria']:6d} {c['virus']:6d} {o.get('train', 0):10d} {o.get('test', 0):9d}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
