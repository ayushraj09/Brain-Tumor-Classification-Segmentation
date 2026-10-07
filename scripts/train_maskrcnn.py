"""Mask R-CNN (ResNet-50 FPN, COCO init) for binary tumor segmentation on the official patient-wise folds
(test = fold k, validation = fold k+1). Healthy scans are included with empty targets. The best epoch is selected by
validation Dice and the score threshold by validation F1.

    python scripts/train_maskrcnn.py --exp seg_cv --folds 1 2 3 4 5 --epochs 20 --batch-size 8 --select dice
"""

from __future__ import annotations

import argparse
import random
import time

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from common import CKPT_DIR, DATA_DIR, DEVICE, RESULTS_DIR, save_json

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mat_file(mat_path: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(mat_path, "r") as f:
        cj = f["cjdata"]
        img = np.array(cj["image"], dtype=np.float32).T
        mask = np.array(cj["tumorMask"], dtype=np.uint8).T
    lo, hi = img.min(), img.max()
    img = ((img - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(img, np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), mask


def mask_to_bbox(mask: np.ndarray):
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return [float(x0), float(y0), float(max(x1, x0 + 1)), float(max(y1, y0 + 1))]


class BrainTumorDataset(torch.utils.data.Dataset):
    """Tumor samples (.mat → box + mask) and no-tumor samples (JPEG → empty target)."""

    def __init__(self, samples: list[dict], train: bool = False):
        self.samples, self.train = samples, train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        if s["type"] == "tumor":
            img, mask = load_mat_file(s["path"])
            box = mask_to_bbox(mask)
        else:
            img, mask, box = np.array(Image.open(s["path"]).convert("RGB")), None, None
        image = TF.to_tensor(img)
        h, w = image.shape[1:]
        if box is None:
            target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64),
                      "masks": torch.zeros((0, h, w), dtype=torch.uint8)}
        else:
            target = {"boxes": torch.tensor([box]), "labels": torch.tensor([1]),
                      "masks": torch.from_numpy(mask[None].copy())}
            # joint h-flip (p=0.5) and v-flip (p=0.3) of image, mask and box, tumor samples only
            if self.train and random.random() < 0.5:
                image, target["masks"] = image.flip(-1), target["masks"].flip(-1)
                target["boxes"][:, [0, 2]] = w - target["boxes"][:, [2, 0]]
            if self.train and random.random() < 0.3:
                image, target["masks"] = image.flip(-2), target["masks"].flip(-2)
                target["boxes"][:, [1, 3]] = h - target["boxes"][:, [3, 1]]
        target["image_id"] = torch.tensor([idx])
        return image, target


def collate(batch):
    return tuple(zip(*batch))


def build_maskrcnn(num_classes: int = 2) -> torch.nn.Module:
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes)
    return model


# ── Splits ────────────────────────────────────────────────────────────────────


def patient_split(test_fold: int):
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    val_fold = test_fold % 5 + 1

    def rows(mask):
        return [{"type": "tumor" if r.source == "figshare" else "no_tumor",
                 "path": r.mat if r.source == "figshare" else r.path} for r in meta[mask].itertuples()]

    return (rows(~meta.fold.isin([test_fold, val_fold])), rows(meta.fold == val_fold), rows(meta.fold == test_fold))


# ── Evaluation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def collect_predictions(model, dataset) -> list[dict]:
    """Best-scoring instance per image (score, binary mask) and the GT mask."""
    model.eval()
    out = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=8, collate_fn=collate)
    for images, targets in loader:
        preds = model([im.to(DEVICE) for im in images])
        for p, t in zip(preds, targets):
            gt = t["masks"][0].numpy().astype(bool) if t["masks"].numel() else None
            if len(p["scores"]):
                j = int(p["scores"].argmax())
                out.append({"gt": gt, "score": float(p["scores"][j]), "pred": (p["masks"][j, 0] > 0.5).cpu().numpy()})
            else:
                out.append({"gt": gt, "score": 0.0, "pred": None})
    return out


def evaluate(preds: list[dict], thr: float) -> dict:
    tp = fn = tn = fp = 0
    ious, dices, ious_hit = [], [], []
    for r in preds:
        detected = r["score"] >= thr
        if r["gt"] is not None:
            if detected:
                tp += 1
                inter, union = np.logical_and(r["pred"], r["gt"]).sum(), np.logical_or(r["pred"], r["gt"]).sum()
                ious.append(inter / union if union else 0.0)
                ious_hit.append(ious[-1])
                dices.append(2 * inter / (r["pred"].sum() + r["gt"].sum()))
            else:
                fn += 1
                ious.append(0.0)
                dices.append(0.0)
        else:
            tn, fp = (tn + 1, fp) if not detected else (tn, fp + 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(threshold=thr, tp=tp, fn=fn, tn=tn, fp=fp, precision=precision, recall=recall,
                specificity=tn / (tn + fp) if tn + fp else 0.0,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                mean_iou=float(np.mean(ious)), mean_dice=float(np.mean(dices)),
                mean_iou_detected=float(np.mean(ious_hit)) if ious_hit else 0.0)


# ── Training ──────────────────────────────────────────────────────────────────


def run(exp: str, splits, args) -> dict:
    seed_all(args.seed)
    train_s, val_s, test_s = splits
    train_ds, val_ds, test_ds = BrainTumorDataset(train_s, True), BrainTumorDataset(val_s), BrainTumorDataset(test_s)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                         pin_memory=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=8, collate_fn=collate)
    model = build_maskrcnn().to(DEVICE)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    ckpt = CKPT_DIR / exp / "mask_rcnn_tumor_best.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr = 0.0
        for images, targets in loader:
            images = [im.to(DEVICE) for im in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr += loss.item()
        sched.step()
        with torch.no_grad():  # validation loss (detection losses need train mode; no gradients)
            vl = sum(sum(model([i.to(DEVICE) for i in ims], [{k: v.to(DEVICE) for k, v in t.items()} for t in tg]).values()).item()
                     for ims, tg in val_loader) / len(val_loader)
        rec = dict(epoch=epoch, train_loss=tr / len(loader), val_loss=vl, sec=time.time() - t0)
        if args.select == "dice":
            rec["val_dice@0.5"] = evaluate(collect_predictions(model, val_ds), 0.5)["mean_dice"]
            score = rec["val_dice@0.5"]
        else:
            score = -vl
        history.append(rec)
        if best is None or score > best[0]:
            best = (score, epoch)
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "num_classes": 2, "val_loss": vl}, ckpt)
        print(f"[{exp}] {rec}", flush=True)

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model_state"])
    val_preds, test_preds = collect_predictions(model, val_ds), collect_predictions(model, test_ds)
    # Threshold chosen on validation only (max F1, ties → higher Dice); test reported at that and at a fixed 0.4
    val_sweep = [evaluate(val_preds, t) for t in THRESHOLDS]
    tuned = max(val_sweep, key=lambda r: (round(r["f1"], 4), r["mean_dice"]))["threshold"]
    result = dict(exp=exp, best_epoch=best[1], history=history, val_sweep=val_sweep, tuned_threshold=tuned,
                  test_at_0_4=evaluate(test_preds, 0.4), test_tuned=evaluate(test_preds, tuned),
                  test_sweep=[evaluate(test_preds, t) for t in THRESHOLDS],
                  n=dict(train=len(train_s), val=len(val_s), test=len(test_s)))
    print(f"[{exp}] TEST@0.4 {result['test_at_0_4']}\n[{exp}] TEST@{tuned} {result['test_tuned']}", flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--split", default="patient", choices=["patient"], help="kept for explicit command lines")
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--select", choices=["val_loss", "dice"], default="dice")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    runs = {f"{args.exp}/fold{k}": patient_split(k) for k in args.folds}
    for exp, splits in runs.items():
        out = RESULTS_DIR / exp / "maskrcnn.json"
        if not out.exists():
            save_json(run(exp, splits, args), out)


if __name__ == "__main__":
    main()
