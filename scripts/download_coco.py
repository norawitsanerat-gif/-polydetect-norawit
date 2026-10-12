"""Select COCO 2017 images that contain person/car/bus and download them.

Usage:
    python scripts/download_coco.py            # both splits
    python scripts/download_coco.py train      # single split
"""

import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "coco_raw"

# coco category_id -> class name
CLASSES = {1: "person", 3: "car", 6: "bus"}

SPLITS = {
    "train": {
        "ann": "instances_train2017.json",
        "url": "http://images.cocodataset.org/train2017/",
        "dir": RAW / "images" / "train",
        "min_per_class": 160,
        "max_images": 600,
    },
    "val": {
        "ann": "instances_val2017.json",
        "url": "http://images.cocodataset.org/val2017/",
        "dir": RAW / "images" / "val",
        "min_per_class": 50,
        "max_images": 200,
    },
}


def select_images(split, ann_path, min_per_class, max_images, seed=42):
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    for cid in CLASSES:
        assert cats[cid] == CLASSES[cid], f"category {cid} is {cats[cid]}, expected {CLASSES[cid]}"

    per_image = {}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid not in CLASSES or ann.get("iscrowd", 0):
            continue
        per_image.setdefault(ann["image_id"], []).append(cid)

    images = {img["id"]: img for img in coco["images"]}

    rng = random.Random(seed)
    candidates = list(per_image.keys())
    rng.shuffle(candidates)

    bus_imgs = [i for i in candidates if 6 in per_image[i]]
    rng.shuffle(bus_imgs)
    rest = [i for i in candidates if 6 not in per_image[i]]

    chosen, totals = [], {c: 0 for c in CLASSES}
    chosen_set = set()

    def add(iid):
        chosen.append(iid)
        chosen_set.add(iid)
        for cid in per_image[iid]:
            totals[cid] += 1

    # pass 1: satisfy quota of the rarest class (bus) first
    for iid in bus_imgs:
        if totals[6] >= min_per_class:
            break
        add(iid)

    # pass 2: fill up to max_images with the remaining pool (diverse, shuffled)
    for iid in candidates:
        if len(chosen) >= max_images:
            break
        if iid not in chosen_set:
            add(iid)

    # pass 3: top up any class still under quota
    for iid in candidates:
        if all(totals[c] >= min_per_class for c in CLASSES):
            break
        if iid not in chosen_set:
            add(iid)

    rows = []
    for iid in chosen:
        cids = per_image[iid]
        counts = {CLASSES[c]: cids.count(c) for c in CLASSES}
        rows.append({"id": iid, "file_name": images[iid]["file_name"], "counts": counts})

    return rows, totals


def download_one(url, path, retries=3):
    if path.exists() and path.stat().st_size > 5000:
        return "skip"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "coco-subset/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as out:
                out.write(r.read())
            if path.stat().st_size < 5000:
                raise IOError("file too small")
            return "ok"
        except Exception:
            if attempt == retries - 1:
                return "fail"
            time.sleep(1 + attempt * 2)
    return "fail"


def run_split(name, cfg):
    ann_path = RAW / cfg["ann"]
    if not ann_path.exists():
        print(f"[{name}] missing {ann_path}")
        return False

    rows, totals = select_images(name, ann_path, cfg["min_per_class"], cfg["max_images"])
    cfg["dir"].mkdir(parents=True, exist_ok=True)

    keep = {r["file_name"] for r in rows}
    for stale in cfg["dir"].glob("*.jpg"):
        if stale.name not in keep:
            stale.unlink()

    print(f"[{name}] selected {len(rows)} images")
    print(f"[{name}] instance counts: " + ", ".join(
        f"{CLASSES[c]}={totals[c]} (target {cfg['min_per_class']})" for c in CLASSES))
    if any(totals[c] < cfg["min_per_class"] for c in CLASSES):
        print(f"[{name}] WARNING: below target - raise max_images")

    ok = fail = skip = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(download_one, cfg["url"] + r["file_name"], cfg["dir"] / r["file_name"]): r
            for r in rows
        }
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            ok += res == "ok"
            skip += res == "skip"
            fail += res == "fail"
            if i % 50 == 0 or i == len(futs):
                print(f"[{name}] {i}/{len(futs)} (ok={ok} skip={skip} fail={fail})")

    manifest = {
        "split": name,
        "classes": [CLASSES[c] for c in sorted(CLASSES)],
        "counts": {CLASSES[c]: totals[c] for c in CLASSES},
        "images": rows,
        "downloaded": {"ok": ok, "skip": skip, "fail": fail},
    }
    out = RAW / f"manifest_{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[{name}] manifest -> {out}")
    return fail == 0


def main():
    names = sys.argv[1:] or list(SPLITS)
    all_ok = True
    for name in names:
        all_ok &= run_split(name, SPLITS[name])
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
