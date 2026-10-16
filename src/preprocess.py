#!/usr/bin/env python3
"""Convert every image in the split file to a fixed-size 8-bit grayscale PNG and write data/manifest.csv."""
import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image


def convert(job):
    src, dst, size = job
    try:
        with Image.open(src) as im:
            im = im.convert("L").resize((size, size), Image.LANCZOS)
            im.save(dst, format="PNG", compress_level=1)
        return None
    except Exception as e:  # noqa: BLE001 - report and continue
        return f"{src}: {e!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--raw-root", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    with open(args.splits, newline="") as f:
        rows = list(csv.DictReader(f))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for i, r in enumerate(rows):
        r["proc_path"] = str(out / f"img_{i:05d}.png")
        jobs.append((Path(args.raw_root) / r["path"], Path(r["proc_path"]), args.size))

    t0 = time.time()
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, err in enumerate(ex.map(convert, jobs, chunksize=16), 1):
            if err:
                errors.append(err)
            if n % 1000 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} converted ({time.time() - t0:.0f}s)", flush=True)

    if errors:
        print("\n".join(errors[:20]), file=sys.stderr)
        sys.exit(f"{len(errors)} images failed to convert")

    with open(args.manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.manifest} ({len(rows)} rows, {args.size}x{args.size} grayscale PNG)")


if __name__ == "__main__":
    main()
