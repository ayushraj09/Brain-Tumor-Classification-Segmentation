"""Tumor boxes predicted by the patient-level Mask R-CNN of each fold, for region-of-interest (ROI) classification.

For CV fold k the segmenter ``checkpoints/seg_cv/fold{k}`` was trained on the three training folds only, so its boxes
on fold k (test) and fold k+1 (validation) are out-of-sample — the same images the fold-k classifier is validated
and tested on. Output: ``results/boxes/fold{k}.json`` mapping image path → [x1, y1, x2, y2, score] of the best
detection, or null when nothing is detected.

    python scripts/predict_boxes.py --folds 1 2 3 4 5
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from common import CKPT_DIR, DATA_DIR, DEVICE, RESULTS_DIR, save_json
from train_maskrcnn import build_maskrcnn


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    for k in args.folds:
        model = build_maskrcnn()
        model.load_state_dict(torch.load(CKPT_DIR / "seg_cv" / f"fold{k}" / "mask_rcnn_tumor_best.pth",
                                         map_location="cpu")["model_state"])
        model = model.to(DEVICE).eval()
        rows = meta[meta.fold.isin([k, k % 5 + 1])]
        boxes: dict[str, list | None] = {}
        paths = rows.path.tolist()
        for i in range(0, len(paths), args.batch_size):
            batch = paths[i : i + args.batch_size]
            preds = model([TF.to_tensor(Image.open(p).convert("RGB")).to(DEVICE) for p in batch])
            for p, pr in zip(batch, preds):
                if len(pr["scores"]):
                    j = int(pr["scores"].argmax())
                    boxes[p] = [*map(float, pr["boxes"][j].tolist()), float(pr["scores"][j])]
                else:
                    boxes[p] = None
        save_json(boxes, RESULTS_DIR / "boxes" / f"fold{k}.json")
        print(f"fold {k}: {len(boxes)} images, {sum(b is None for b in boxes.values())} without detection", flush=True)


if __name__ == "__main__":
    main()
