from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import scipy.io
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_b3
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


ROOT_DIR = Path(__file__).resolve().parent
# Both models were trained with the official patient-wise fold 1 held out (scripts/train_classifier.py
# --exp cv4_attn, scripts/train_maskrcnn.py --exp seg_cv), so fold-1 images are unseen by both stages.
CLASSIFIER_CKPT = ROOT_DIR / "checkpoints" / "classifier" / "EfficientNet_B3_gradcam_guided.pt"
CLASSIFIER_SIZE = 300
MASKRCNN_CKPT = ROOT_DIR / "checkpoints" / "mask_rcnn_tumor_best.pth"
HELD_OUT_FOLD = 1
DATASET_DIR = ROOT_DIR / "Brain-Tumor-Dataset"
METADATA_CSV = DATASET_DIR / "metadata.csv"
TUMOR_MASK_DIR = DATASET_DIR / "Tumor-Mask"

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]
MAT_LABEL_MAP = {1: "meningioma", 2: "glioma", 3: "pituitary_tumor"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_classifier_model(num_classes: int = 4) -> torch.nn.Module:
    model = efficientnet_b3(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_maskrcnn(num_classes: int) -> torch.nn.Module:
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_feat_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat_box, num_classes)

    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_feat_mask,
        dim_reduced=256,
        num_classes=num_classes,
    )
    return model


@st.cache_resource
def load_classifier() -> torch.nn.Module:
    if not CLASSIFIER_CKPT.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {CLASSIFIER_CKPT}")

    model = build_classifier_model(num_classes=len(CLASS_NAMES)).to(DEVICE)
    checkpoint = torch.load(CLASSIFIER_CKPT, map_location=DEVICE)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@st.cache_resource
def load_segmenter() -> tuple[torch.nn.Module, dict[str, Any]]:
    if not MASKRCNN_CKPT.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {MASKRCNN_CKPT}")

    checkpoint = torch.load(MASKRCNN_CKPT, map_location=DEVICE)
    num_classes = checkpoint.get("num_classes", 2) if isinstance(checkpoint, dict) else 2
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint

    model = build_maskrcnn(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint if isinstance(checkpoint, dict) else {}


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    min_v, max_v = float(image.min()), float(image.max())
    if max_v <= min_v:
        return np.zeros_like(image, dtype=np.uint8)
    return ((image - min_v) / (max_v - min_v) * 255.0).clip(0, 255).astype(np.uint8)


def _decode_pid(pid_array: np.ndarray) -> str:
    flat = np.asarray(pid_array).reshape(-1)
    if flat.size == 0:
        return ""
    if np.issubdtype(flat.dtype, np.number):
        ints = [int(x) for x in flat]
        if all(0 <= x < 256 for x in ints):
            try:
                text = "".join(chr(x) for x in ints).strip("\x00 ").strip()
                if text:
                    return text
            except Exception:
                pass
        return "".join(str(int(x)) for x in ints)
    return str(flat.tolist())


def _parse_h5_mat(payload: bytes) -> dict[str, Any]:
    with h5py.File(io.BytesIO(payload), "r") as f:
        cj = f["cjdata"]
        img_raw = np.array(cj["image"], dtype=np.float32).T
        mask = np.array(cj["tumorMask"], dtype=np.uint8).T
        label = int(np.array(cj["label"]).squeeze())
        pid = _decode_pid(np.array(cj["PID"]))
        border = np.array(cj["tumorBorder"], dtype=np.float32).reshape(-1)

    gray = _normalize_to_uint8(img_raw)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return {
        "image_rgb": rgb,
        "tumor_mask": (mask > 0).astype(np.uint8),
        "tumor_border": border,
        "label": label,
        "pid": pid,
    }


def _parse_scipy_mat(payload: bytes) -> dict[str, Any]:
    mat = scipy.io.loadmat(io.BytesIO(payload), squeeze_me=True, struct_as_record=False)
    cj = mat["cjdata"]

    img_raw = np.asarray(cj.image, dtype=np.float32)
    mask = np.asarray(cj.tumorMask, dtype=np.uint8)
    label = int(np.asarray(cj.label).squeeze())
    pid = _decode_pid(np.asarray(cj.PID))
    border = np.asarray(cj.tumorBorder, dtype=np.float32).reshape(-1)

    gray = _normalize_to_uint8(img_raw)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return {
        "image_rgb": rgb,
        "tumor_mask": (mask > 0).astype(np.uint8),
        "tumor_border": border,
        "label": label,
        "pid": pid,
    }


def load_mat_case(payload: bytes) -> dict[str, Any]:
    try:
        data = _parse_h5_mat(payload)
    except Exception:
        data = _parse_scipy_mat(payload)
    data["source_type"] = "mat"
    data["has_ground_truth"] = True
    return data


def _resolve_jpg_gt_mask(
    image_filename: str,
    source_path: str | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """
    Resolve JPEG GT mask from Tumor-Mask directory.
    Returns (mask_binary_or_none, gt_class_or_none).
    """
    if not TUMOR_MASK_DIR.exists():
        return None, None

    fname = Path(image_filename).name
    candidate: Path | None = None
    gt_class: str | None = None

    # If source path is known from Training/<class>/file.jpg, try direct mapping first.
    if source_path is not None:
        p = Path(source_path)
        cls = p.parent.name
        direct = TUMOR_MASK_DIR / cls / fname
        if direct.exists():
            candidate = direct
            gt_class = cls

    # Fallback: recursive search by filename in Tumor-Mask/*
    if candidate is None:
        hits = sorted(TUMOR_MASK_DIR.glob(f"**/{fname}"))
        if hits:
            candidate = hits[0]
            gt_class = candidate.parent.name

    if candidate is None:
        return None, None

    # Load as color because mask files may be colormap images
    # (e.g., yellow tumor on violet background).
    mask_bgr = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
    if mask_bgr is None:
        return None, None

    # 1) Primary path: detect yellow foreground (ignore violet background).
    hsv = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2HSV)
    # OpenCV HSV ranges: H:[0..179], S:[0..255], V:[0..255]
    lower_yellow = np.array([18, 80, 80], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_bin = (yellow_mask > 0).astype(np.uint8)

    # 2) Fallback: if no yellow detected, use classic binary threshold.
    # Useful when mask files are true grayscale/binary.
    if int(mask_bin.sum()) == 0:
        mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        mask_bin = (thr > 0).astype(np.uint8)

    return mask_bin, gt_class


def load_image_case(payload: bytes, image_filename: str, source_path: str | None = None) -> dict[str, Any]:
    raw = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Unable to decode uploaded image.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gt_mask, gt_class = _resolve_jpg_gt_mask(image_filename=image_filename, source_path=source_path)
    if gt_mask is not None and gt_mask.shape != rgb.shape[:2]:
        gt_mask = cv2.resize(
            gt_mask.astype(np.uint8),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)
    return {
        "image_rgb": rgb,
        "tumor_mask": gt_mask,
        "tumor_border": np.array([], dtype=np.float32),
        "label": None,
        "gt_class_name": gt_class,
        "pid": "",
        "source_type": "image",
        "has_ground_truth": gt_mask is not None,
    }


def classifier_input(image_rgb: np.ndarray) -> torch.Tensor:
    """Same as the training eval pipeline (albumentations Resize = cv2 bilinear, then ImageNet normalisation)."""
    x = cv2.resize(image_rgb, (CLASSIFIER_SIZE, CLASSIFIER_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    x = (x / 255.0 - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)


def predict_tumor_class(model: torch.nn.Module, image_rgb: np.ndarray) -> tuple[str, float, np.ndarray, np.ndarray]:
    """Class probabilities and the Grad-CAM (last conv block) of the predicted class, resized to the input."""
    store: dict[str, torch.Tensor] = {}
    handle = model.features[-1].register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    try:
        logits = model(classifier_input(image_rgb))
        idx = int(logits.argmax(1))
        grads = torch.autograd.grad(logits[0, idx], store["a"])[0]
    finally:
        handle.remove()
    cam = F.relu((grads.mean((2, 3), keepdim=True) * store["a"]).sum(1))[0].detach().cpu().numpy()
    cam = cv2.resize(cam, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    cam = cam / cam.max() if cam.max() > 0 else cam
    probs = torch.softmax(logits.detach(), dim=1).squeeze(0).cpu().numpy()
    return CLASS_NAMES[idx], float(probs[idx]), probs, cam


def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (image_rgb.astype(np.float32) * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    score_thresh: float = 0.5,
) -> dict[str, Any]:
    image_tensor = T.ToTensor()(image_rgb).to(DEVICE)
    pred = model([image_tensor])[0]
    keep = pred["scores"] >= score_thresh
    return {
        "boxes": pred["boxes"][keep].cpu().numpy(),
        "scores": pred["scores"][keep].cpu().numpy(),
        "labels": pred["labels"][keep].cpu().numpy(),
        "masks": pred["masks"][keep].cpu().numpy(),
        "tumor_detected": bool(keep.any().item()),
    }


def overlay_binary_mask(image_rgb: np.ndarray, mask_binary: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    out = image_rgb.copy().astype(np.float32)
    red = np.zeros_like(out)
    red[..., 0] = 255.0
    m = mask_binary.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * red[m]
    return out.clip(0, 255).astype(np.uint8)


def draw_bbox(image_rgb: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    out = image_rgb.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def overlay_gt_pred_masks(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray | None = None,
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Overlay masks for comparison:
      - GT mask in blue
      - Predicted mask in red
    """
    out = image_rgb.copy().astype(np.float32)

    gt_color = np.zeros_like(out)
    gt_color[..., 2] = 255.0  # blue in RGB
    gt_bool = gt_mask.astype(bool)
    out[gt_bool] = (1.0 - alpha) * out[gt_bool] + alpha * gt_color[gt_bool]

    if pred_mask is not None:
        pred_color = np.zeros_like(out)
        pred_color[..., 0] = 255.0  # red in RGB
        pred_bool = pred_mask.astype(bool)
        out[pred_bool] = (1.0 - alpha) * out[pred_bool] + alpha * pred_color[pred_bool]

    return out.clip(0, 255).astype(np.uint8)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:  # both empty: perfect agreement
        return 1.0
    return float(inter / union)


def _held_out_rows() -> pd.DataFrame:
    """Held-out (patient-wise fold 1) rows of metadata.csv written by scripts/prepare_data.py."""
    if not METADATA_CSV.exists():
        return pd.DataFrame()
    meta = pd.read_csv(METADATA_CSV)
    meta = meta[meta.fold == HELD_OUT_FOLD]
    # metadata stores absolute paths from the machine that built it; re-root them on this checkout
    def reroot(p: str) -> str:
        parts = Path(p).parts
        return str(DATASET_DIR.joinpath(*parts[parts.index(DATASET_DIR.name) + 1 :])) if p else ""

    for col in ("path", "mat"):
        meta[col] = meta[col].fillna("").map(reroot)
    return meta


@st.cache_data
def get_suggested_test_files() -> dict[str, str]:
    """One held-out JPEG per class."""
    meta = _held_out_rows()
    return {c: g.path.iloc[0] for c, g in meta.groupby("cls") if Path(g.path.iloc[0]).exists()} if len(meta) else {}


@st.cache_data
def get_suggested_mat_files() -> dict[str, str]:
    """One held-out .mat file per tumor class."""
    meta = _held_out_rows()
    if not len(meta):
        return {}
    meta = meta[meta.mat != ""]
    return {c: g.mat.iloc[0] for c, g in meta.groupby("cls") if Path(g.mat.iloc[0]).exists()}


def main() -> None:
    st.set_page_config(page_title="Brain MRI Inference", layout="wide")
    st.title("Brain MRI Tumor Classification + Segmentation")

    st.caption(
        f"Device: `{DEVICE}` | Classifier: `{CLASSIFIER_CKPT}` | Segmenter: `{MASKRCNN_CKPT}`"
    )

    score_thresh = st.sidebar.slider("Mask R-CNN score threshold", 0.05, 0.95, 0.50, 0.01)
    mask_thresh = st.sidebar.slider("Mask binarization threshold", 0.05, 0.95, 0.50, 0.01)

    uploaded = st.file_uploader(
        "Upload `.mat` (tumor dataset) or an image (`.jpg/.jpeg/.png`)",
        type=["mat", "jpg", "jpeg", "png"],
    )
    suggested_jpg_files = get_suggested_test_files()
    suggested_mat_files = get_suggested_mat_files()
    suggested_options_map: dict[str, str] = {}
    if suggested_jpg_files:
        for cls in CLASS_NAMES:
            if cls in suggested_jpg_files:
                suggested_options_map[f"JPG held-out - {cls}"] = suggested_jpg_files[cls]
    if suggested_mat_files:
        for cls in ["meningioma", "glioma", "pituitary_tumor"]:
            if cls in suggested_mat_files:
                suggested_options_map[f"MAT held-out - {cls}"] = suggested_mat_files[cls]

    suggestion_options = ["None"] + list(suggested_options_map.keys())
    selected_suggestion = st.selectbox(
        "Or select a built-in suggested file",
        options=suggestion_options,
        index=0,
    )

    ready_to_run = not (uploaded is None and selected_suggestion == "None")
    run_clicked = st.button("Run Inference", type="primary", disabled=not ready_to_run)

    if not ready_to_run:
        st.info("Upload one file or choose a suggested test file.")
        return
    if not run_clicked:
        st.info("Adjust thresholds/select file, then click **Run Inference**.")
        return

    if uploaded is not None:
        payload = uploaded.getvalue()
        filename = uploaded.name
        suffix = Path(filename).suffix.lower()
    else:
        suggested_path = Path(suggested_options_map[selected_suggestion])
        payload = suggested_path.read_bytes()
        filename = suggested_path.name
        suffix = suggested_path.suffix.lower()

    source_path_str: str | None = None
    if suffix == ".mat":
        case = load_mat_case(payload)
    else:
        if uploaded is None and selected_suggestion != "None":
            source_path_str = str(suggested_path)
        case = load_image_case(payload, image_filename=filename, source_path=source_path_str)

    image_rgb = case["image_rgb"]
    gt_mask = case["tumor_mask"]

    with st.spinner("Loading models..."):
        clf = load_classifier()
        seg, seg_meta = load_segmenter()

    pred_class, pred_conf, class_probs, cam = predict_tumor_class(clf, image_rgb)
    seg_pred = predict_mask(seg, image_rgb, score_thresh=score_thresh)

    pid = case["pid"] or "N/A"
    col_a, col_b = st.columns(2)
    col_a.metric("Predicted tumor type", pred_class)
    col_b.metric("Classifier confidence", f"{pred_conf:.3f}")

    st.subheader("Classification probabilities")
    st.bar_chart({name: float(p) for name, p in zip(CLASS_NAMES, class_probs)})

    vis_cols = st.columns(3)
    vis_cols[0].image(image_rgb, caption="Input image", channels="RGB", use_container_width=True)
    vis_cols[2].image(
        overlay_cam(image_rgb, cam),
        caption=f"Grad-CAM for '{pred_class}' (where the classifier looked)",
        channels="RGB",
        use_container_width=True,
    )

    if seg_pred["tumor_detected"]:
        best_idx = int(np.argmax(seg_pred["scores"]))
        best_score = float(seg_pred["scores"][best_idx])
        pred_mask = (seg_pred["masks"][best_idx, 0] > mask_thresh).astype(np.uint8)
        pred_box = seg_pred["boxes"][best_idx]

        overlay = overlay_binary_mask(image_rgb, pred_mask, alpha=0.35)
        overlay = draw_bbox(overlay, pred_box)
        vis_cols[1].image(
            overlay,
            caption=f"Predicted mask + box (score={best_score:.3f})",
            channels="RGB",
            use_container_width=True,
        )
        st.success("Tumor detected by Mask R-CNN.")
    else:
        vis_cols[1].warning("No mask passed the score threshold.")

    with st.expander("Ground truth (optional)"):
        st.write(f"Input file: `{filename}`")
        st.write(f"Source type: `{case['source_type']}`")
        if case["source_type"] == "mat":
            gt_label = case["label"]
            gt_label_name = MAT_LABEL_MAP.get(gt_label, f"unknown({gt_label})")
            st.write(f"PID: `{pid}`")
            st.write(f"Ground-truth class from `.mat`: `{gt_label_name}` (label `{gt_label}`)")
            st.caption("Legend: **Blue = GT mask**, **Red = Predicted mask**")
            if seg_pred["tumor_detected"]:
                iou = mask_iou(gt_mask, pred_mask)
                comp = overlay_gt_pred_masks(image_rgb, gt_mask, pred_mask, alpha=0.35)
                st.image(comp, caption="GT (blue) vs Prediction (red)", channels="RGB", use_container_width=True)
                st.write(f"IoU (predicted best mask vs GT `tumorMask`): `{iou:.4f}`")
            else:
                comp = overlay_gt_pred_masks(image_rgb, gt_mask, None, alpha=0.35)
                st.image(comp, caption="GT only (blue) — no predicted mask", channels="RGB", use_container_width=True)
                st.write("IoU: N/A (no predicted mask passed threshold)")
        else:
            if case["has_ground_truth"] and gt_mask is not None:
                gt_name = case.get("gt_class_name") or "unknown"
                st.write(f"Ground-truth class from JPG mask path: `{gt_name}`")
                st.caption("Legend: **Blue = GT mask**, **Red = Predicted mask**")
                if seg_pred["tumor_detected"]:
                    iou = mask_iou(gt_mask, pred_mask)
                    comp = overlay_gt_pred_masks(image_rgb, gt_mask, pred_mask, alpha=0.35)
                    st.image(comp, caption="GT (blue) vs Prediction (red)", channels="RGB", use_container_width=True)
                    st.write(f"IoU (predicted best mask vs JPG GT mask): `{iou:.4f}`")
                else:
                    comp = overlay_gt_pred_masks(image_rgb, gt_mask, None, alpha=0.35)
                    st.image(comp, caption="GT only (blue) — no predicted mask", channels="RGB", use_container_width=True)
                    st.write("IoU: N/A (no predicted mask passed threshold)")
            else:
                st.info(
                    "No ground-truth mask found for this image in `Brain-Tumor-Dataset/Tumor-Mask`."
                )

    with st.expander("Model input requirements (checked)"):
        st.markdown(
            f"- **EfficientNet-B3 (Grad-CAM-guided, patient-level CV fold {HELD_OUT_FOLD})**: RGB image -> "
            f"`Resize({CLASSIFIER_SIZE},{CLASSIFIER_SIZE})` (bilinear) -> "
            f"`Normalize(mean={IMAGENET_MEAN}, std={IMAGENET_STD})` -> shape `(1,3,{CLASSIFIER_SIZE},{CLASSIFIER_SIZE})`.\n"
            "- **Mask R-CNN**: RGB image -> `ToTensor()` only (`float` in `[0,1]`) -> "
            "shape `(3,H,W)` passed as a list `[tensor]`; no manual resize/normalize in app."
        )

if __name__ == "__main__":
    main()
