"""Run predictions on val images with both trained models and save results.

Outputs:
    output/predict/rec/   - annotated images from detect model
    output/predict/poly/  - annotated images from segment model
    output/predict/predictions.csv - table of all detections
"""

import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VAL_IMG = ROOT / "dataset" / "coco_raw" / "images" / "val"
OUT = ROOT / "output" / "predict"
N_SAMPLES = 12  # images to annotate & save

def main():
    (OUT / "rec").mkdir(parents=True, exist_ok=True)
    (OUT / "poly").mkdir(parents=True, exist_ok=True)

    imgs = sorted(VAL_IMG.glob("*.jpg"))
    sample = imgs[:N_SAMPLES]

    rows = []
    # Rec model
    print("=== Rec (detect) ===", flush=True)
    rec = YOLO(str(ROOT / "runs" / "detect" / "coco_rec" / "weights" / "best.pt"))
    for p in imgs:
        r = rec.predict(str(p), imgsz=640, device=0, workers=2, verbose=False)[0]
        for b in r.boxes:
            cls = r.names[int(b.cls[0])]
            conf = float(b.conf[0])
            xyxy = [round(float(x), 2) for x in b.xyxy[0]]
            rows.append({"model": "rec", "image": p.name, "class": cls,
                         "conf": round(conf, 4), "xyxy": str(xyxy)})
    # save annotated for sample
    for p in sample:
        r = rec.predict(str(p), imgsz=640, device=0, workers=2, verbose=False)[0]
        r.save(str(OUT / "rec" / p.name))

    # Poly model
    print("=== Poly (segment) ===", flush=True)
    poly = YOLO(str(ROOT / "runs" / "segment" / "coco_poly" / "weights" / "best.pt"))
    for p in imgs:
        r = poly.predict(str(p), imgsz=640, device=0, workers=2, verbose=False)[0]
        for b in r.boxes:
            cls = r.names[int(b.cls[0])]
            conf = float(b.conf[0])
            xyxy = [round(float(x), 2) for x in b.xyxy[0]]
            rows.append({"model": "poly", "image": p.name, "class": cls,
                         "conf": round(conf, 4), "xyxy": str(xyxy)})
    for p in sample:
        r = poly.predict(str(p), imgsz=640, device=0, workers=2, verbose=False)[0]
        r.save(str(OUT / "poly" / p.name))

    # write csv
    with open(OUT / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "image", "class", "conf", "xyxy"])
        w.writeheader()
        w.writerows(rows)

    # summary
    from collections import Counter
    for m in ("rec", "poly"):
        c = Counter(r["class"] for r in rows if r["model"] == m)
        confs = [r["conf"] for r in rows if r["model"] == m]
        avg_conf = sum(confs) / len(confs) if confs else 0
        print(f"[{m}] detections={len(confs)} avg_conf={avg_conf:.3f} "
              f"classes={dict(c)}")

    print(f"saved {N_SAMPLES} sample images per model -> {OUT}")
    print(f"predictions.csv -> {OUT / 'predictions.csv'}")

if __name__ == "__main__":
    main()
