"""Run a fixed YOLO detector on a pre-fused image dir against LLVIP labels."""
import argparse
import glob
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fuse', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--imgsz', type=int, default=640)
    args = ap.parse_args()

    val_dir = os.path.join(args.fuse.rstrip('/') + f'_{args.tag}_val')
    img_dir = os.path.join(val_dir, 'images')
    lab_dir = os.path.join(val_dir, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    n_img = 0
    for p in glob.glob(os.path.join(args.fuse, '*.png')):
        stem = os.path.splitext(os.path.basename(p))[0]
        lab = os.path.join(args.labels, stem + '.txt')
        if not os.path.exists(lab):
            continue
        shutil.copy(p, os.path.join(img_dir, os.path.basename(p)))
        shutil.copy(lab, os.path.join(lab_dir, os.path.basename(lab)))
        n_img += 1
    print(f'[{args.tag}] {n_img} images matched')
    if n_img == 0:
        return

    yaml_path = os.path.join(val_dir, 'data.yaml')
    with open(yaml_path, 'w') as fh:
        fh.write(f"path: {val_dir}\ntrain: images\nval: images\nnc: 1\nnames: {{0: person}}\n")

    from ultralytics import YOLO
    det = YOLO(args.model)
    m = det.val(data=yaml_path, imgsz=args.imgsz, batch=16, device=0,
                verbose=False, project=val_dir, name='val')
    print(f'[{args.tag}] mAP50={getattr(m.box, "map50", 0.0):.4f}  '
          f'mAP50-95={getattr(m.box, "map", 0.0):.4f}')


if __name__ == '__main__':
    main()
