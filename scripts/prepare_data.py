"""Build the dataset folders used by every experiment.

Inputs (``<root>/raw``):
    mats/*.mat               Figshare brain tumor dataset (Cheng et al. 2015), 3064 slices, 233 patients
    cvind.mat                official patient-wise 5-fold CV indices shipped with the Figshare dataset
    sartaj/Training/no_tumor 395 healthy scans (Bhuvaji et al., Kaggle "Brain Tumor Classification (MRI)")

Outputs (``<root>/Brain-Tumor-Dataset``):
    Training/<class>/<id>.jpg         classification images (ImageFolder layout)
    Brain-Tumor-Images-Mat-Files/     symlinks to the .mat files (segmentation)
    Tumor-Mask/<class>/<id>.png       binary GT masks (Grad-CAM localisation metrics, Streamlit GT overlay)
    metadata.csv                      one row per image: path, class, patient id, official CV fold, mask path
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

MAT_LABEL_MAP = {1: "meningioma", 2: "glioma", 3: "pituitary_tumor"}


def decode_pid(arr: np.ndarray) -> str:
    """PID is stored as a uint16 char array in MATLAB v7.3 files."""
    return "".join(chr(int(c)) for c in np.asarray(arr).reshape(-1)).strip("\x00 ").strip()


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Min-max rescaling, identical to the MATLAB snippet in the Figshare README."""
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    return np.zeros_like(img, dtype=np.uint8) if hi <= lo else np.uint8(255.0 / (hi - lo) * (img - lo))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/mri"))
    parser.add_argument("--no-tumor-folds-seed", type=int, default=42)
    args = parser.parse_args()

    raw = args.root / "raw"
    out = args.root / "Brain-Tumor-Dataset"
    train_dir, mask_dir, mat_link_dir = out / "Training", out / "Tumor-Mask", out / "Brain-Tumor-Images-Mat-Files"
    for d in [*(train_dir / c for c in [*MAT_LABEL_MAP.values(), "no_tumor"]), *(mask_dir / c for c in MAT_LABEL_MAP.values()), mat_link_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with h5py.File(raw / "cvind.mat", "r") as f:
        cvind = np.array(f["cvind"]).reshape(-1).astype(int)  # fold (1..5) of mat id i+1
    assert cvind.shape[0] == 3064, cvind.shape

    rows = []
    for mat_path in sorted((raw / "mats").glob("*.mat"), key=lambda p: int(p.stem)):
        mid = int(mat_path.stem)
        with h5py.File(mat_path, "r") as f:
            cj = f["cjdata"]
            image = to_uint8(np.array(cj["image"]).T)
            mask = (np.array(cj["tumorMask"]).T > 0).astype(np.uint8)
            label = int(np.array(cj["label"]).squeeze())
            pid = decode_pid(np.array(cj["PID"]))
        cls = MAT_LABEL_MAP[label]
        img_path, msk_path = train_dir / cls / f"{mid}.jpg", mask_dir / cls / f"{mid}.png"
        cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(msk_path), mask * 255)
        link = mat_link_dir / mat_path.name
        if not link.exists():
            link.symlink_to(mat_path.resolve())
        rows.append(
            dict(path=str(img_path), cls=cls, pid=f"p{pid}", fold=int(cvind[mid - 1]), mask=str(msk_path),
                 mat=str(link), source="figshare", height=image.shape[0], width=image.shape[1])
        )

    # Healthy scans come from a different source without patient ids: each image is its own group and is
    # assigned to one of 5 folds with a fixed seed, stratified by simple round-robin over a shuffled order.
    nt_src = sorted((raw / "sartaj" / "Training" / "no_tumor").glob("*"))
    rng = np.random.default_rng(args.no_tumor_folds_seed)
    order = rng.permutation(len(nt_src))
    for rank, i in enumerate(order):
        src = nt_src[i]
        dst = train_dir / "no_tumor" / src.name
        shutil.copy2(src, dst)
        h, w = cv2.imread(str(dst), cv2.IMREAD_GRAYSCALE).shape
        rows.append(
            dict(path=str(dst), cls="no_tumor", pid=f"nt_{src.stem}", fold=int(rank % 5) + 1, mask="",
                 mat="", source="sartaj", height=h, width=w)
        )

    meta = pd.DataFrame(rows)
    meta.to_csv(out / "metadata.csv", index=False)
    print(meta.groupby("cls").size())
    print("patients:", meta[meta.source == "figshare"].pid.nunique())
    print(pd.crosstab(meta.cls, meta.fold))
    # Sanity check: the official folds are patient-disjoint.
    fig = meta[meta.source == "figshare"]
    assert (fig.groupby("pid").fold.nunique() == 1).all(), "official folds are not patient-wise"


if __name__ == "__main__":
    main()
