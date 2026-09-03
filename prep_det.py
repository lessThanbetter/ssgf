"""Build a small, uploadable LLVIP detection subset (VOC XML -> YOLO format).

Reads directly from the LLVIP zip, samples train/test pairs, and writes
``visible/ infrared/ labels/`` per split so a YOLO detector can be trained and
evaluated on fused images.

Usage:
    python prep_det.py --zip C:/Users/12137/Downloads/LLVIP/LLVIP.zip \
        --dest data/det_subset --train 1500 --test 300
"""

from __future__ import annotations

import argparse, os, random, zipfile, xml.etree.ElementTree as ET
import shutil


def parse_voc(xml_bytes):
    root = ET.fromstring(xml_bytes)
    fname = root.findtext("filename")
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name").strip()
        b = obj.find("bndbox")
        x1 = float(b.findtext("xmin")); y1 = float(b.findtext("ymin"))
        x2 = float(b.findtext("xmax")); y2 = float(b.findtext("ymax"))
        boxes.append((name, x1, y1, x2, y2))
    return fname, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--dest", default="data/det_subset")
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--test", type=int, default=300)
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    xmls = [n for n in zf.namelist() if n.endswith(".xml")]
    ann = {}
    for x in xmls:
        try:
            fname, boxes = parse_voc(zf.read(x))
        except Exception:
            continue
        if fname and boxes:
            ann[fname] = boxes
    print(f"[prep] parsed {len(ann)} annotated images")

    # image names that exist in both visible+infrared (use visible to find)
    vis = [n for n in zf.namelist() if n.startswith("LLVIP/visible/train/")]
    vis_test = [n for n in zf.namelist() if n.startswith("LLVIP/visible/test/")]
    def base(p): return os.path.basename(p)
    train_names = [base(p) for p in vis if base(p) in ann]
    test_names = [base(p) for p in vis_test if base(p) in ann]
    random.seed(0)
    random.shuffle(train_names); train_names = train_names[: args.train]
    random.shuffle(test_names); test_names = test_names[: args.test]
    print(f"[prep] sampled train={len(train_names)} test={len(test_names)}")

    def rel(path_from_zip):  # -> (split, file)
        for split, names in (("train", train_names), ("test", test_names)):
            if os.path.basename(path_from_zip) in names and os.path.basename(path_from_zip) in set(names):
                return split
        return None

    import numpy as np
    from PIL import Image
    written = 0
    for split, names in (("train", train_names), ("test", test_names)):
        for d in ("visible", "infrared", "labels"):
            os.makedirs(os.path.join(args.dest, split, d), exist_ok=True)
        for name in names:
            boxes = ann[name]
            for mod in ("visible", "infrared"):
                zpath = f"LLVIP/{mod}/{split}/{name}"
                data = zf.read(zpath)
                with open(os.path.join(args.dest, split, mod, name), "wb") as fh:
                    fh.write(data)
            with Image.open(zf.open(f"LLVIP/visible/{split}/{name}")) as im:
                W, H = im.size
            with open(os.path.join(args.dest, split, "labels", name.replace(".jpg", ".txt")), "w") as fh:
                for objname, x1, y1, x2, y2 in boxes:
                    cls = 0 if "person" in objname.lower() else 0
                    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                    w, h = (x2 - x1) / W, (y2 - y1) / H
                    fh.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            written += 1
    total_mb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(args.dest) for f in fs) / 1e6
    print(f"[prep] wrote {written} pairs, {total_mb:.0f} MB to {args.dest}")


if __name__ == "__main__":
    main()
