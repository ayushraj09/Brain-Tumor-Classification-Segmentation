"""Grad-CAM explainability analysis of trained classifiers.

For every test image of a run (``results/<tag>/<model>.json``) the Grad-CAM of the predicted class is computed at
the last convolutional block and scored against the Figshare ground-truth tumor mask:

* pointing game      – CAM maximum lies inside the tumor mask dilated by ``tol`` (Zhang et al., 2018)
* energy_in_tumor    – fraction of CAM energy inside the (undilated) tumor mask (energy pointing game, Wang 2020)
* energy_in_context  – same, mask dilated by ``tol`` (peritumoral context, Cheng et al. 2015)
* cam_iou            – IoU between CAM > 0.5·max and the tumor mask
* energy_in_brain    – fraction of CAM energy inside the head (Otsu foreground); low values flag shortcut learning
                       on background / acquisition artefacts — measured on *all* images, incl. no_tumor

    python scripts/gradcam_analysis.py --tags cv4_improved/fold1 cv4_improved/fold2 ... --out gradcam/cv4_improved
"""

from __future__ import annotations

import argparse
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common import *  # noqa: E402,F403

CNN_MODELS = ["AlexNet", "InceptionV3", "EfficientNet_B3"]


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model, self.store = model, {}
        for m in model.modules():
            if hasattr(m, "inplace"):
                m.inplace = False
        layer.register_forward_hook(lambda m, i, o: self.store.__setitem__("a", o))

    def __call__(self, x: torch.Tensor, target: torch.Tensor | None = None):
        self.model.eval()
        x = x.requires_grad_(False)
        with torch.enable_grad():
            logits = split_logits(self.model(x))[0]
            target = logits.argmax(1) if target is None else target
            a = self.store["a"]
            g = torch.autograd.grad(logits.gather(1, target[:, None]).sum(), a)[0]
        cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True)).detach()
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / (cam.flatten(1).max(1).values[:, None, None] + 1e-8)
        return cam.cpu().numpy(), torch.softmax(logits.detach(), 1).cpu().numpy()


def brain_mask(gray: np.ndarray) -> np.ndarray:
    _, m = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return m.astype(bool)


def score(cam: np.ndarray, tumor: np.ndarray | None, brain: np.ndarray, tol: int) -> dict:
    e = cam.sum() + 1e-8
    # an all-zero Grad-CAM (all gradient-weighted activations negative) carries no spatial explanation
    out = {"cam_empty": float(cam.max() <= 0), "energy_in_brain": float(cam[brain].sum() / e)}
    if tumor is not None and tumor.any():
        ctx = cv2.dilate(tumor.astype(np.uint8), np.ones((tol, tol), np.uint8)).astype(bool)
        peak = np.unravel_index(cam.argmax(), cam.shape)
        hot = cam > 0.5
        out.update(pointing=float(ctx[peak]), energy_in_tumor=float(cam[tumor].sum() / e),
                   energy_in_context=float(cam[ctx].sum() / e),
                   cam_iou=float((hot & tumor).sum() / ((hot | tumor).sum() + 1e-8)),
                   tumor_area=float(tumor.mean()))
    return out


def analyse(tag: str, model_name: str, n_fig: int, fig_dir: Path, tol_frac: float) -> list[dict]:
    res = json.loads((RESULTS_DIR / tag / f"{model_name}.json").read_text())
    class_names, recipe = res["classes"], Recipe(**res["recipe"])
    size = recipe.size_for(model_name)
    df = load_samples(class_names)
    model = build_model(model_name, len(class_names), size)
    model.load_state_dict(torch.load(CKPT_DIR / tag / f"{model_name}.pt", map_location="cpu"))
    model = model.to(DEVICE)
    cam_fn = GradCAM(model, gradcam_layer(model, model_name))
    eval_tf = torchvision_eval_transform(size) if recipe.name == "baseline" else albu_eval(size)
    ds = MRIDataset(df, res["test_idx"], eval_tf, size, with_mask=True)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8)
    tol = max(3, int(round(tol_frac * size)) | 1)

    rows, shown = [], {c: 0 for c in class_names}
    fig_items = []
    pos = 0
    for x, y, m in loader:
        cams, probs = cam_fn(x.to(DEVICE))
        for b in range(len(y)):
            i = res["test_idx"][pos]
            pos += 1
            row = df.iloc[i]
            gray = cv2.resize(cv2.imread(row.path, cv2.IMREAD_GRAYSCALE), (size, size), interpolation=cv2.INTER_AREA)
            tumor = m[b].numpy().astype(bool)
            pred = int(probs[b].argmax())
            rec = dict(tag=tag, model=model_name, path=row.path, cls=row.cls, pred=class_names[pred],
                       correct=bool(pred == int(y[b])), confidence=float(probs[b].max()),
                       **score(cams[b], tumor if tumor.any() else None, brain_mask(gray), tol))
            rows.append(rec)
            if shown[row.cls] < n_fig:
                shown[row.cls] += 1
                fig_items.append((gray, tumor, cams[b], rec))
    plot_grid(fig_items, fig_dir / f"{tag.replace('/', '_')}_{model_name}.png", f"{tag} — {model_name}")
    del cam_fn, model
    torch.cuda.empty_cache()
    return rows


def plot_grid(items, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = 4
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (gray, tumor, cam, rec) in zip(np.atleast_1d(axes).ravel(), items):
        ax.imshow(gray, cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.4)
        if tumor.any():
            ax.contour(tumor, levels=[0.5], colors="white", linewidths=1)
        mark = "✓" if rec["correct"] else "✗"
        extra = f"\nE_tumor={rec['energy_in_tumor']:.2f} hit={int(rec['pointing'])}" if "pointing" in rec else ""
        ax.set_title(f"{rec['cls']} → {rec['pred']} {mark}{extra}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    tum = df[df.pointing.notna()]
    agg = tum.groupby("model").agg(pointing=("pointing", "mean"), energy_in_tumor=("energy_in_tumor", "mean"),
                                   energy_in_context=("energy_in_context", "mean"), cam_iou=("cam_iou", "mean"))
    agg["chance_energy"] = tum.groupby("model").tumor_area.mean()  # energy_in_tumor of a uniform map
    agg["cam_empty"] = df.groupby("model").cam_empty.mean()
    agg["energy_in_brain_all"] = df.groupby("model").energy_in_brain.mean()
    agg["energy_in_brain_no_tumor"] = df[df.cls == "no_tumor"].groupby("model").energy_in_brain.mean()
    agg["pointing_correct"] = tum[tum.correct].groupby("model").pointing.mean()
    agg["pointing_wrong"] = tum[~tum.correct].groupby("model").pointing.mean()
    agg["n_wrong_tumor"] = tum[~tum.correct].groupby("model").size()
    return agg.reset_index()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--models", nargs="+", default=CNN_MODELS)
    p.add_argument("--out", required=True)
    p.add_argument("--n-fig", type=int, default=3, help="example images per class per run in the figure grid")
    p.add_argument("--tol-frac", type=float, default=0.07, help="pointing-game tolerance as a fraction of input size")
    args = p.parse_args()
    out = RESULTS_DIR / args.out
    rows = [r for tag in args.tags for name in args.models for r in analyse(tag, name, args.n_fig, out / "figures", args.tol_frac)]
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image.csv", index=False)
    summary = summarise(df)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
