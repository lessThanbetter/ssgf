"""Train an independent detector without the failing OpenCV DataLoader path.

This script uses PIL for letterboxing and calls the Ultralytics DetectionModel
loss directly. It is intended only for the independent YOLOv9t confirmatory
evaluator after the stock Windows DataLoader was found to abort natively.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ultralytics.models.yolo.detect import DetectionTrainer


class DetectionPairs(Dataset):
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        size: int,
        train: bool,
        cache: bool,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.size = size
        self.train = train
        self.stems = sorted(
            path.stem
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not self.stems:
            raise RuntimeError(f"No images in {image_dir}")
        self.cache = (
            [self._read(index) for index in range(len(self.stems))]
            if cache
            else None
        )

    def __len__(self) -> int:
        return len(self.stems)

    def _read(self, index: int):
        stem = self.stems[index]
        image_path = next(
            path
            for suffix in (".jpg", ".jpeg", ".png", ".bmp")
            if (path := self.image_dir / f"{stem}{suffix}").is_file()
        )
        image = Image.open(image_path).convert("RGB")
        labels = []
        label_path = self.label_dir / f"{stem}.txt"
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    labels.append([float(value) for value in parts[:5]])

        width, height = image.size
        scale = self.size / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        image = image.resize(
            (resized_width, resized_height),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        pad_x = (self.size - resized_width) // 2
        pad_y = (self.size - resized_height) // 2
        canvas.paste(image, (pad_x, pad_y))

        boxes = []
        classes = []
        for cls, xc, yc, box_width, box_height in labels:
            x1 = (xc - box_width / 2) * resized_width + pad_x
            y1 = (yc - box_height / 2) * resized_height + pad_y
            x2 = (xc + box_width / 2) * resized_width + pad_x
            y2 = (yc + box_height / 2) * resized_height + pad_y
            boxes.append(
                [
                    (x1 + x2) / (2 * self.size),
                    (y1 + y2) / (2 * self.size),
                    (x2 - x1) / self.size,
                    (y2 - y1) / self.size,
                ]
            )
            classes.append([cls])

        return {
            "img": np.asarray(canvas, dtype=np.uint8),
            "bboxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            "cls": np.asarray(classes, dtype=np.float32).reshape(-1, 1),
            "im_file": str(image_path),
        }

    def __getitem__(self, index: int):
        sample = (
            self.cache[index]
            if self.cache is not None
            else self._read(index)
        )
        image = sample["img"].copy()
        boxes = sample["bboxes"].copy()
        if self.train:
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                boxes[:, 0] = 1.0 - boxes[:, 0]
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            pixels = image.astype(np.float32) * brightness
            channel_mean = pixels.mean(axis=(0, 1), keepdims=True)
            pixels = (pixels - channel_mean) * contrast + channel_mean
            image = np.clip(pixels, 0, 255).astype(np.uint8)
        return {
            "img": image,
            "bboxes": boxes,
            "cls": sample["cls"],
            "im_file": sample["im_file"],
        }


def collate(samples):
    images = torch.from_numpy(
        np.stack([sample["img"] for sample in samples], axis=0)
    ).permute(0, 3, 1, 2)
    boxes = []
    classes = []
    batch_indices = []
    for batch_index, sample in enumerate(samples):
        count = len(sample["bboxes"])
        if count:
            boxes.append(torch.from_numpy(sample["bboxes"]))
            classes.append(torch.from_numpy(sample["cls"]))
            batch_indices.append(torch.full((count,), batch_index, dtype=torch.long))
    return {
        "img": images.float() / 255.0,
        "bboxes": (
            torch.cat(boxes, dim=0)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32)
        ),
        "cls": (
            torch.cat(classes, dim=0)
            if classes
            else torch.zeros((0, 1), dtype=torch.float32)
        ),
        "batch_idx": (
            torch.cat(batch_indices, dim=0)
            if batch_indices
            else torch.zeros((0,), dtype=torch.long)
        ),
        "im_file": [sample["im_file"] for sample in samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    trainer = DetectionTrainer(
        overrides={
            "model": str(args.weights),
            "data": str(args.data),
            "epochs": 1,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": 0,
            "device": 0,
            "plots": False,
            "val": False,
            "amp": False,
            "deterministic": False,
            "project": str(args.project),
            "name": args.name,
            "exist_ok": True,
        }
    )
    trainer.setup_model()
    model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    model.train()

    data = trainer.data
    train_dataset = DetectionPairs(
        Path(data["train"]),
        Path(str(data["path"])) / "labels" / "train",
        args.imgsz,
        train=True,
        cache=not args.no_cache,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-4,
    )
    epochs = 1 if args.smoke else args.epochs
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = min(max(1, 3 * len(train_loader)), total_steps)
    global_step = 0

    def set_learning_rate() -> float:
        if global_step < warmup_steps:
            factor = (global_step + 1) / warmup_steps
            lr = args.lr * factor
        else:
            progress = min(
                1.0,
                (global_step - warmup_steps)
                / max(1, total_steps - warmup_steps),
            )
            lr = args.lr * (
                0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
            )
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    output = args.project / args.name
    weights_dir = output / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    log = []
    start = time.time()
    for epoch in range(epochs):
        totals = []
        for step, batch in enumerate(train_loader):
            batch = {
                key: value.to(trainer.device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            learning_rate = set_learning_rate()
            predictions = model(batch["img"])
            loss, loss_items = model.loss(batch, predictions)
            loss = loss.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            global_step += 1
            totals.append(float(loss.detach()))
            if step % 20 == 0:
                print(
                    f"epoch={epoch + 1}/{epochs} step={step + 1}/{len(train_loader)} "
                    f"loss={float(loss.detach()):.4f} "
                    f"box={float(loss_items.get('box_loss', 0)):.3f} "
                    f"cls={float(loss_items.get('cls_loss', 0)):.3f} "
                    f"dfl={float(loss_items.get('dfl_loss', 0)):.3f} "
                    f"lr={learning_rate:.6g}",
                    flush=True,
                )
            if args.smoke or (
                args.max_steps is not None
                and global_step >= args.max_steps
            ):
                break
        mean_loss = float(np.mean(totals))
        log.append({"epoch": epoch + 1, "loss": mean_loss})
        print(f"epoch={epoch + 1} mean_loss={mean_loss:.4f}", flush=True)
        torch.save(
            {
                "model": model,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "train_args": {
                    "weights": str(args.weights),
                    "data": str(args.data),
                    "epochs": epochs,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "seed": args.seed,
                    "lr": args.lr,
                    "loader": "PIL manual loader",
                },
            },
            weights_dir / "last.pt",
        )
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    (output / "manual_train_log.json").write_text(
        json.dumps(
            {
                "epochs": epochs,
                "seconds": time.time() - start,
                "batches_per_epoch": len(train_loader),
                "log": log,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"WROTE {weights_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
