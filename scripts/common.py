"""Shared data, model, training and evaluation code for the classification experiments.

All experiments use the official patient-wise folds of the Figshare dataset (``metadata.csv``, column ``fold``).
Training options are fields of ``Recipe``; ``RECIPES`` holds the two recipes compared in the report.
"""

from __future__ import annotations

import json
import math
import os
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

ROOT = Path(os.environ.get("MRI_ROOT", "/data/mri"))
DATA_DIR = ROOT / "Brain-Tumor-Dataset"
TRAIN_DIR = DATA_DIR / "Training"
RESULTS_DIR = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]
TUMOR_CLASSES = ["glioma", "meningioma", "pituitary_tumor"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_NAMES = ["MLP", "AlexNet", "InceptionV3", "EfficientNet_B3"]
EXTRA_MODELS = ["ConvNeXt_T", "EfficientNetV2_S"]  # stronger ImageNet backbones for the patient-level study
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Recipe:
    """Training recipe. Defaults = ``baseline`` (resize only, AdamW 1e-4, batch 128, 15 epochs)."""

    name: str = "baseline"
    epochs: int = 15
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    cosine: bool = False
    warmup_epochs: int = 0
    aug: str = "none"  # none | light
    image_size: dict = field(default_factory=lambda: {"InceptionV3": 299})
    aux_loss: float = 0.0  # InceptionV3 auxiliary-head weight
    amp: bool = False
    tta: bool = False  # average logits over the image and its horizontal flip
    select_metric: str = "acc"  # validation metric used to keep the best epoch
    attn_weight: float = 0.0  # Grad-CAM-guided attention loss weight (see train_classifier.py)
    attn_form: str = "one_minus_inside"  # or "outside_ratio" (first formulation; rewards an all-zero CAM)
    ema: float = 0.0  # >0: keep an exponential moving average of the weights and evaluate/save it

    def size_for(self, model_name: str) -> int:
        return self.image_size.get(model_name, 224)


RECIPES = {
    "baseline": Recipe(),
    "improved": Recipe(
        name="improved", epochs=30, batch_size=32, lr=2e-4, weight_decay=0.05, label_smoothing=0.1, cosine=True,
        warmup_epochs=1, aug="light", image_size={"InceptionV3": 299, "EfficientNet_B3": 300}, aux_loss=0.4,
        amp=True, tta=True, select_metric="macro_f1",
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Splits ────────────────────────────────────────────────────────────────────


def load_samples(classes: list[str]) -> pd.DataFrame:
    """ImageFolder-ordered samples joined with metadata.csv (patient ID, official fold, mask path)."""
    folder = datasets.ImageFolder(TRAIN_DIR)
    assert folder.classes == CLASS_NAMES, folder.classes
    df = pd.DataFrame(folder.samples, columns=["path", "label"])
    df["cls"] = [CLASS_NAMES[i] for i in df.label]
    meta = pd.read_csv(DATA_DIR / "metadata.csv").drop(columns="cls")
    df = df.merge(meta, on="path", how="left", validate="one_to_one")
    df = df[df.cls.isin(classes)].reset_index(drop=True)
    df["label"] = [classes.index(c) for c in df.cls]
    return df


def patient_cv_split(df: pd.DataFrame, test_fold: int) -> tuple[list[int], list[int], list[int]]:
    """Official patient-wise folds (cvind.mat): test = fold k, validation = next fold, train = other three."""
    val_fold = test_fold % 5 + 1
    test = df.index[df.fold == test_fold].tolist()
    val = df.index[df.fold == val_fold].tolist()
    train = df.index[~df.fold.isin([test_fold, val_fold])].tolist()
    return train, val, test


# ── Data ──────────────────────────────────────────────────────────────────────


def torchvision_eval_transform(size: int) -> transforms.Compose:
    """Transform of the ``baseline`` recipe (PIL resize, bilinear + antialias)."""
    return transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


def light_train_aug(size: int) -> A.Compose:
    """MRI-plausible augmentation: geometry + intensity only (no colour jitter on grayscale, no elastic warps)."""
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def albu_eval(size: int) -> A.Compose:
    return A.Compose([A.Resize(size, size), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


def crop_roi(image: np.ndarray, mask: np.ndarray, box, context: float, jitter: bool):
    """Square crop centred on ``box`` (x1, y1, x2, y2) with side = context × longest box side (at least a quarter of
    the image), zero-padded at the borders. ``jitter`` randomly rescales (×0.8–1.25) and shifts (±10 %) the crop."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box[:4]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(max(x2 - x1, y2 - y1) * context, 0.25 * min(h, w))
    if jitter:
        side *= random.uniform(0.8, 1.25)
        cx += random.uniform(-0.1, 0.1) * side
        cy += random.uniform(-0.1, 0.1) * side
    side = min(side, max(h, w))
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = int(round(side))
    pad = max(0, -left, -top, left + s - w, top + s - h)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    left, top = left + pad, top + pad
    return image[top : top + s, left : left + s], mask[top : top + s, left : left + s]


class MRIDataset(Dataset):
    """Returns (image, label[, mask]). ``mask`` is the resized tumor mask (zeros for no_tumor / unknown); it is only
    used by the attention-guided loss and by the Grad-CAM localisation metrics.

    ``roi`` switches to tumor-region crops (see ``crop_roi``): ``"gt"`` crops around the ground-truth mask with
    random jitter (training), ``"pred"`` around the Mask R-CNN box in ``boxes[path]`` (validation / test). Images
    without a box (no tumor, or nothing detected) are used whole."""

    def __init__(self, df: pd.DataFrame, indices: list[int], transform, size: int, with_mask: bool = False,
                 roi: str | None = None, boxes: dict | None = None, context: float = 2.0):
        self.df, self.indices, self.transform, self.size, self.with_mask = df, indices, transform, size, with_mask
        self.roi, self.boxes, self.context = roi, boxes or {}, context
        self.is_albu = isinstance(transform, A.Compose)
        assert self.is_albu or roi is None, "ROI crops are implemented for the albumentations pipelines"

    def __len__(self) -> int:
        return len(self.indices)

    def _mask(self, row) -> np.ndarray:
        if isinstance(row["mask"], str) and row["mask"]:
            return (np.array(Image.open(row["mask"])) > 0).astype(np.uint8)
        return None

    def _box(self, row, mask: np.ndarray):
        if self.roi == "pred":
            return self.boxes.get(row.path)
        if self.roi == "gt" and mask.any():
            ys, xs = np.where(mask)
            return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        return None

    def __getitem__(self, i: int):
        row = self.df.iloc[self.indices[i]]
        image = Image.open(row.path).convert("RGB")
        if not self.is_albu:
            x = self.transform(image)
            if not self.with_mask:
                return x, int(row.label)
            m = self._mask(row)
            m = np.zeros((self.size, self.size), np.uint8) if m is None else np.array(
                Image.fromarray(m).resize((self.size, self.size), Image.NEAREST))
            return x, int(row.label), torch.from_numpy(m)
        image = np.array(image)
        m = self._mask(row) if (self.with_mask or self.roi) else None
        m = np.zeros(image.shape[:2], np.uint8) if m is None else m
        if self.roi:
            box = self._box(row, m)
            if box is not None:
                image, m = crop_roi(image, m, box, self.context, jitter=self.roi == "gt")
        if not self.with_mask:
            return self.transform(image=image)["image"], int(row.label)
        out = self.transform(image=image, mask=m)
        return out["image"], int(row.label), torch.as_tensor(out["mask"]).to(torch.uint8)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int = 42) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    # drop_last on the training loader avoids a size-1 final batch, which BatchNorm1d (MLP) cannot train on
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=min(12, os.cpu_count() or 2),
                      pin_memory=torch.cuda.is_available(), generator=g, persistent_workers=True, drop_last=shuffle)


# ── Models ────────────────────────────────────────────────────────────────────


class BrainMLP(nn.Module):
    def __init__(self, num_classes: int, image_size: int = 224):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * image_size * image_size, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model(name: str, num_classes: int, image_size: int = 224) -> nn.Module:
    if name == "MLP":
        return BrainMLP(num_classes, image_size)
    if name == "AlexNet":
        m = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
        return m
    if name == "InceptionV3":
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, num_classes)
        return m
    if name == "EfficientNet_B3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if name == "ConvNeXt_T":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    if name == "EfficientNetV2_S":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    raise ValueError(name)


def gradcam_layer(model: nn.Module, name: str) -> nn.Module | None:
    """Last convolutional block (post-activation) — the standard Grad-CAM target. The MLP has none."""
    return {
        "AlexNet": lambda: model.features[11],
        "InceptionV3": lambda: model.Mixed_7c,
        "EfficientNet_B3": lambda: model.features[-1],
        "ConvNeXt_T": lambda: model.features[-1],
        "EfficientNetV2_S": lambda: model.features[-1],
    }.get(name, lambda: None)()


# ── Training / evaluation ─────────────────────────────────────────────────────


def split_logits(output):
    """InceptionV3 returns (logits, aux_logits) in train mode."""
    if hasattr(output, "logits"):
        return output.logits, getattr(output, "aux_logits", None)
    return output, None


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, tta: bool = False, amp: bool = False):
    model.eval()
    probs, targets = [], []
    for batch in loader:
        x, y = batch[0].to(DEVICE, non_blocking=True), batch[1]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = split_logits(model(x))[0].float()
            if tta:
                logits = (logits + split_logits(model(torch.flip(x, dims=[3])))[0].float()) / 2
        probs.append(torch.softmax(logits, 1).cpu())
        targets.append(y)
    return torch.cat(targets).numpy(), torch.cat(probs).numpy()


def metrics(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    y_pred = probs.argmax(1)
    r = classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names,
                              output_dict=True, zero_division=0)
    w = r["weighted avg"]
    return {
        "accuracy": r["accuracy"], "precision": w["precision"], "recall": w["recall"], "f1": w["f1-score"],
        "macro_f1": r["macro avg"]["f1-score"],
        "per_class": {c: {k: r[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in class_names},
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist(),
    }


def lr_lambda(recipe: Recipe, steps_per_epoch: int):
    """Per-step LR multiplier: optional linear warm-up, then cosine decay to 1e-6 (evaluated per epoch when there
    is no warm-up, i.e. ``CosineAnnealingLR(T_max=epochs, eta_min=1e-6)`` stepped once per epoch)."""
    total, warm = recipe.epochs * steps_per_epoch, recipe.warmup_epochs * steps_per_epoch
    floor = 1e-6 / recipe.lr

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        if not recipe.cosine:
            return 1.0
        if warm == 0:
            progress = (step // steps_per_epoch) / recipe.epochs
        else:
            progress = (step - warm) / max(1, total - warm)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return f


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


__all__ = [n for n in dir() if not n.startswith("_")] + ["asdict", "deepcopy"]
