"""Build YOLO detect (bbox) and segment (polygon) datasets from the COCO subset.

Both datasets use the exact same images and the exact same objects:
the bbox in coco_rec is computed from the very polygons stored in coco_poly,
so the two formats are directly comparable.

Outputs:
    dataset/coco_rec/   images/{train,val} + labels/{train,val}
    dataset/coco_poly/  images/{train,val} + labels/{train,val}
    custom_rec.yaml
    custom_poly.yaml
    output/phase2/class_counts.json
    output/phase2/vis_*.jpg
"""

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "coco_raw"
REC_ROOT = ROOT / "dataset" / "coco_rec"
POLY_ROOT = ROOT / "dataset" / "coco_poly"
OUT = ROOT / "output" / "phase2"

CLASSES = ["person", "car", "bus"]
COCO2CLS = {1: 0, 3: 1, 6: 2}
MIN_AREA = 400.0  # px^2 - drop slivers that cannot be learned at 640px
MIN_TRAIN = 100  # requirement: >= 100 objects per label in train
SPLITS = ["train", "val"]
COLORS_BGR = [(60, 180, 75), (255, 140, 0), (230, 60, 60)]

HYPER = """\
# Hyperparameters
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.2
translate: 0.1
scale: 0.75
shear: 0.0
perspective: 0.001
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
copy_paste_mode: mixup
auto_augment: randaugment
erasing: 0.4
"""


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def clean_polygon(flat, w, h):
    """flat = [x1,y1,x2,y2,...] -> normalized points or None."""
    if not flat or len(flat) % 2 or len(flat) < 6:
        return None
    pts = np.asarray(flat, dtype=np.float64).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, w)
    pts[:, 1] = np.clip(pts[:, 1], 0, h)
    keep = np.concatenate(([True], np.any(np.diff(pts, axis=0) != 0, axis=1)))
    pts = pts[keep]
    if len(pts) < 3:
        return None
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    if (x1 - x0) * (y1 - y0) < MIN_AREA:
        return None
    return pts


def process_split(split):
    with open(RAW / f"manifest_{split}.json", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(RAW / f"instances_{split}2017.json", encoding="utf-8") as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] in COCO2CLS and not a.get("iscrowd", 0):
            anns_by_img[a["image_id"]].append(a)
    del coco

    stats = {
        "images_selected": len(manifest["images"]),
        "images_written": 0,
        "images_skipped": 0,
        "objects_dropped_small": 0,
        "rec_objects": Counter(),
        "poly_objects": Counter(),
        "poly_lines": Counter(),
    }
    skipped = []

    for item in manifest["images"]:
        img = images[item["id"]]
        w, h = img["width"], img["height"]
        src = RAW / "images" / split / item["file_name"]
        rec_lines, poly_lines, dropped = [], [], 0

        for a in anns_by_img.get(item["id"], []):
            cls = COCO2CLS[a["category_id"]]
            polys = [clean_polygon(p, w, h) for p in (a.get("segmentation") or [])]
            polys = [p for p in polys if p is not None]

            if polys:
                all_pts = np.vstack(polys)
                x0, y0 = all_pts.min(axis=0)
                x1, y1 = all_pts.max(axis=0)
            else:
                x, y, bw, bh = a["bbox"]
                x0, y0, x1, y1 = x, y, x + bw, y + bh
                rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
                if (x1 - x0) * (y1 - y0) < MIN_AREA:
                    polys = []
                else:
                    polys = [rect]

            if not polys or (x1 - x0) * (y1 - y0) < MIN_AREA or x1 - x0 < 2 or y1 - y0 < 2:
                dropped += 1
                continue

            rec_lines.append(
                f"{cls} {((x0 + x1) / 2 / w):.6f} {((y0 + y1) / 2 / h):.6f} "
                f"{((x1 - x0) / w):.6f} {((y1 - y0) / h):.6f}"
            )
            stats["rec_objects"][cls] += 1
            stats["poly_objects"][cls] += 1
            for p in polys:
                coords = np.empty(p.size, dtype=np.float64)
                coords[0::2] = p[:, 0] / w
                coords[1::2] = p[:, 1] / h
                poly_lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in coords))
                stats["poly_lines"][cls] += 1

        if not rec_lines:
            stats["images_skipped"] += 1
            skipped.append(item["file_name"])
            continue

        stats["images_written"] += 1
        stats["objects_dropped_small"] += dropped

        for root, lines in ((REC_ROOT, rec_lines), (POLY_ROOT, poly_lines)):
            link_or_copy(src, root / "images" / split / item["file_name"])
            with open(root / "labels" / split / (Path(item["file_name"]).stem + ".txt"),
                      "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    stats["skipped_files"] = skipped
    return stats


def check_labels(root: Path, split: str, polygon: bool):
    """Every image has a label; every coordinate inside [0,1]."""
    imgs = sorted((root / "images" / split).glob("*.jpg"))
    bad = []
    for im in imgs:
        lf = root / "labels" / split / (im.stem + ".txt")
        if not lf.exists():
            bad.append(f"missing label {lf.name}")
            continue
        for ln, line in enumerate(lf.read_text().splitlines(), 1):
            parts = line.split()
            vals = [float(v) for v in parts[1:]]
            if polygon and len(vals) < 6:
                bad.append(f"{lf.name}:{ln} polygon <3 pts")
            if not polygon and len(vals) != 4:
                bad.append(f"{lf.name}:{ln} bbox !=4 values")
            if any(v < -0.001 or v > 1.001 for v in vals):
                bad.append(f"{lf.name}:{ln} out of range")
    return len(imgs), bad


def draw_samples():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        stems = sorted(
            p.stem for p in (POLY_ROOT / "labels" / split).glob("*.txt")
            if p.read_text().strip()
        )[:3]
        for stem in stems:
            img = cv2.imread(str(RAW / "images" / split / (stem + ".jpg")))
            vis_rec = img.copy()
            vis_poly = img.copy()
            h, w = img.shape[:2]

            for line in (REC_ROOT / "labels" / split / (stem + ".txt")).read_text().splitlines():
                c, cx, cy, bw, bh = line.split()
                c = int(c)
                x0, y0 = int((float(cx) - float(bw) / 2) * w), int((float(cy) - float(bh) / 2) * h)
                x1, y1 = int((float(cx) + float(bw) / 2) * w), int((float(cy) + float(bh) / 2) * h)
                cv2.rectangle(vis_rec, (x0, y0), (x1, y1), COLORS_BGR[c], 2)
                cv2.putText(vis_rec, CLASSES[c], (x0, max(14, y0 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS_BGR[c], 2, cv2.LINE_AA)

            for line in (POLY_ROOT / "labels" / split / (stem + ".txt")).read_text().splitlines():
                parts = line.split()
                c = int(parts[0])
                vals = np.array([float(v) for v in parts[1:]]).reshape(-1, 2)
                pts = (vals * [w, h]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis_poly, [pts], True, COLORS_BGR[c], 2)
                overlay = vis_poly.copy()
                cv2.fillPoly(overlay, [pts], COLORS_BGR[c])
                cv2.addWeighted(overlay, 0.25, vis_poly, 0.75, 0, vis_poly)
                x0, y0 = pts[:, 0, 0].min(), pts[:, 0, 1].min()
                cv2.putText(vis_poly, CLASSES[c], (int(x0), max(14, int(y0) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS_BGR[c], 2, cv2.LINE_AA)

            sep = np.full((h, 6, 3), 255, np.uint8)
            panel = np.hstack([vis_rec, sep, vis_poly])
            cv2.imwrite(str(OUT / f"vis_{split}_{stem}.jpg"), panel)


def write_yaml(name: str, rel_path: str):
    text = (
        f"path: {rel_path}\n\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/val\n\n"
        f"nc: {len(CLASSES)}\n\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
        + "\n" + HYPER
    )
    (ROOT / name).write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_stats = {}

    for split in SPLITS:
        for root in (REC_ROOT, POLY_ROOT):
            for sub in ("images", "labels"):
                (root / sub / split).mkdir(parents=True, exist_ok=True)
        st = process_split(split)
        all_stats[split] = st
        print(f"[{split}] selected={st['images_selected']} written={st['images_written']} "
              f"skipped={st['images_skipped']} dropped_small={st['objects_dropped_small']}")
        print(f"[{split}] rec objects   : " + ", ".join(
            f"{CLASSES[c]}={st['rec_objects'][c]}" for c in range(len(CLASSES))))
        print(f"[{split}] poly objects  : " + ", ".join(
            f"{CLASSES[c]}={st['poly_objects'][c]}" for c in range(len(CLASSES))))
        print(f"[{split}] poly lines    : " + ", ".join(
            f"{CLASSES[c]}={st['poly_lines'][c]}" for c in range(len(CLASSES))))
        if st["skipped_files"]:
            print(f"[{split}] skipped files: {st['skipped_files']}")

    write_yaml("custom_rec.yaml", "./dataset/coco_rec")
    write_yaml("custom_poly.yaml", "./dataset/coco_poly")
    print("wrote custom_rec.yaml / custom_poly.yaml")

    failures = []
    for root, polygon, name in ((REC_ROOT, False, "coco_rec"), (POLY_ROOT, True, "coco_poly")):
        for split in SPLITS:
            n_imgs, bad = check_labels(root, split, polygon)
            print(f"[check] {name}/{split}: {n_imgs} images, problems={len(bad)}")
            failures += [f"{name}/{split}: {b}" for b in bad[:10]]
        train_counts = all_stats["train"]["rec_objects" if not polygon else "poly_objects"]
        for c in range(len(CLASSES)):
            if train_counts[c] < MIN_TRAIN:
                failures.append(f"{name}/train {CLASSES[c]}={train_counts[c]} < {MIN_TRAIN}")

    draw_samples()
    print(f"visualizations -> {OUT}")

    with open(OUT / "class_counts.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                s: {
                    k: (dict(v) if isinstance(v, Counter) else v)
                    for k, v in all_stats[s].items()
                }
                for s in SPLITS
            },
            f, indent=2)

    if failures:
        print("FAIL:")
        for m in failures:
            print("  -", m)
        raise SystemExit(1)
    print(f"PASS: every train label >= {MIN_TRAIN} objects in both datasets, all labels valid")


if __name__ == "__main__":
    main()
