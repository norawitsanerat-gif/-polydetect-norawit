"""Train the Rec (detect) and Poly (segment) models.

Usage:
    python scripts/train.py detect [epochs]
    python scripts/train.py segment [epochs]
    python scripts/train.py both [epochs]
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "detect": {
        "model": ROOT / "yolo26n.pt",
        "data": ROOT / "custom_rec.yaml",
        "project": ROOT / "runs" / "detect",
        "name": "coco_rec",
    },
    "segment": {
        "model": ROOT / "yolo26n-seg.pt",
        "data": ROOT / "custom_poly.yaml",
        "project": ROOT / "runs" / "segment",
        "name": "coco_poly",
    },
}

KEY_COLS = {
    "metrics/precision(B)": "P",
    "metrics/recall(B)": "R",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "metrics/precision(M)": "P_mask",
    "metrics/recall(M)": "R_mask",
    "metrics/mAP50(M)": "mAP50_mask",
    "metrics/mAP50-95(M)": "mAP50-95_mask",
}


def print_final_metrics(csv_path: Path):
    if not csv_path.exists():
        print("no results.csv")
        return
    import csv

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    last = rows[-1]
    print("--- final epoch metrics ---")
    for col, label in KEY_COLS.items():
        if col in last and last[col] not in ("", None):
            print(f"  {label:16s} {float(last[col]):.4f}")
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0) or 0))
    print(f"  best epoch      {best.get('epoch', '?')} "
          f"(mAP50={float(best.get('metrics/mAP50(B)', 0)):.4f})")


def train(which: str, epochs: int):
    cfg = TASKS[which]
    print(f"=== [{which}] model={cfg['model'].name} data={cfg['data'].name} "
          f"epochs={epochs} ===", flush=True)
    model = YOLO(str(cfg["model"]))
    t0 = time.time()
    model.train(
        data=str(cfg["data"]),
        epochs=epochs,
        imgsz=640,
        batch=-1,
        device=0,
        workers=2,
        seed=0,
        deterministic=True,
        project=str(cfg["project"]),
        name=cfg["name"],
        exist_ok=True,
        plots=True,
        val=True,
    )
    mins = (time.time() - t0) / 60
    best = cfg["project"] / cfg["name"] / "weights" / "best.pt"
    print(f"=== [{which}] finished in {mins:.1f} min ===", flush=True)
    print(f"best.pt: {best} exists={best.exists()} "
          f"size={best.stat().st_size if best.exists() else 0} bytes")
    print_final_metrics(cfg["project"] / cfg["name"] / "results.csv")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    if which not in ("detect", "segment", "both"):
        raise SystemExit("usage: train.py [detect|segment|both] [epochs]")
    tasks = ["detect", "segment"] if which == "both" else [which]
    for t in tasks:
        train(t, epochs)


if __name__ == "__main__":
    main()
