"""Figures for report/report.tex, built from the files in results/ (no GPU needed).

    python scripts/make_report_figures.py --results results --out report/figures
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical slots 1-3 (light surface) and text/grid tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODELS = ["AlexNet", "InceptionV3", "EfficientNet_B3", "ENS"]
LABELS = {"AlexNet": "AlexNet", "InceptionV3": "InceptionV3", "EfficientNet_B3": "EfficientNet-B3", "ENS": "Ensemble"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def fold_stats(results: Path, exp: str) -> dict[str, tuple[float, float]]:
    """Mean and std of test accuracy over folds, incl. the probability-averaging ensemble."""
    runs: dict[str, dict[str, dict]] = {}
    for p in (results / exp).glob("fold*/*.json"):
        r = json.loads(p.read_text())
        runs.setdefault(p.parent.name, {})[r["model"]] = r
    accs: dict[str, list[float]] = {}
    for fold in runs.values():
        for m, r in fold.items():
            accs.setdefault(m, []).append(r["test"]["accuracy"])
        if all(m in fold for m in MODELS[:3]):
            y = np.array(fold["AlexNet"]["test_true"])
            p = np.mean([np.array(fold[m]["test_probs"]) for m in MODELS[:3]], axis=0)
            accs.setdefault("ENS", []).append(float((p.argmax(1) == y).mean()))
    return {m: (100 * np.mean(v), 100 * np.std(v, ddof=1)) for m, v in accs.items()}


def single_stats(results: Path, exp: str) -> dict[str, float]:
    out = {}
    for p in (results / exp).glob("*.json"):
        r = json.loads(p.read_text())
        out[r["model"]] = 100 * r["test"]["accuracy"]
    return out


def dot_whisker(ax, groups: list[dict], names: list[str], models: list[str]) -> None:
    x = np.arange(len(models))
    offsets = np.linspace(-0.22, 0.22, len(groups))
    for k, (g, name) in enumerate(zip(groups, names)):
        xs = [x[i] + offsets[k] for i, m in enumerate(models) if m in g]
        mu = [g[m][0] if isinstance(g[m], tuple) else g[m] for m in models if m in g]
        sd = [g[m][1] if isinstance(g[m], tuple) else 0 for m in models if m in g]
        ax.errorbar(xs, mu, yerr=sd, fmt="o", ms=5, color=SERIES[k], ecolor=SERIES[k], elinewidth=1.2, capsize=0,
                    label=name, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xticks(x, [LABELS[m] for m in models])


def fig_accuracy(results: Path, out: Path) -> None:
    groups = [fold_stats(results, e) for e in ("cv3_baseline", "cv3_improved", "cv3_attn")]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    dot_whisker(ax, groups, ["baseline recipe", "improved recipe", "improved + Grad-CAM guidance"], MODELS)
    ax.axhline(97.3, color=TEXT_2, lw=0.8)
    ax.text(-0.45, 97.45, "Díaz-Pernas et al. (2021): 97.3 %", ha="left", va="bottom", color=TEXT_2, fontsize=8)
    ax.set_ylabel("Test accuracy (%), mean ± std over 5 folds")
    ax.set_ylim(85, 100)
    ax.set_xlim(-0.5, 3.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    fig.savefig(out / "accuracy_cv3.pdf")
    plt.close(fig)


def fig_progression(results: Path, out: Path) -> None:
    """Test accuracy (3 classes, patient-level 5-fold) at each step of the method, read from results/."""
    sel = results / "selection"

    def from_sel(name: str) -> tuple[float, float]:
        t = 100 * np.array(json.loads((sel / name).read_text())["test_folds"])
        return t.mean(), t.std(ddof=1)

    steps = [
        ("Baseline\nrecipe", fold_stats(results, "cv3_baseline")["InceptionV3"]),
        ("Improved\nrecipe", fold_stats(results, "cv3_improved")["EfficientNet_B3"]),
        ("+ Grad-CAM\nguidance", fold_stats(results, "p3_full_effb3")["EfficientNet_B3"]),
        ("+ validation-\nselected ensemble", from_sel("sel_groups_val.json")),
        ("+ refit on 4 folds\n(final)", from_sel("refit_ensemble.json")),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    x = np.arange(len(steps))
    mu = [m for _, (m, _) in steps]
    sd = [s for _, (_, s) in steps]
    ax.errorbar(x, mu, yerr=sd, fmt="o", ms=6, color=SERIES[0], elinewidth=1.2, capsize=0,
                markeredgecolor="white", markeredgewidth=0.8)
    for xi, m in zip(x, mu):
        ax.text(xi + 0.12, m, f"{m:.2f}", va="center", fontsize=8, color=TEXT)
    for y, label in [(97.3, "Díaz-Pernas et al. 2021 (97.3)"), (98.0, "Deepak & Ameer 2019 (98.0)")]:
        ax.axhline(y, color=TEXT_2, lw=0.8)
        ax.text(-0.45, y + 0.12, label, fontsize=7.5, color=TEXT_2)
    ax.set_xticks(x, [n for n, _ in steps], fontsize=8)
    ax.set_xlim(-0.5, len(steps) - 0.3)
    ax.set_ylim(90, 99.5)
    ax.set_ylabel("Test accuracy (%), mean ± std")
    fig.savefig(out / "progression.pdf")
    fig.savefig(out / "progression.png", dpi=160)  # for the README
    plt.close(fig)


def fig_gradcam(results: Path, out: Path) -> None:
    exps = [("cv4_baseline", "baseline recipe"), ("cv4_improved", "improved recipe"), ("cv4_attn", "+ Grad-CAM guidance")]
    tabs = {e: pd.read_csv(results / "gradcam" / e / "summary.csv").set_index("model") for e, _ in exps}
    models = MODELS[:3]
    metrics = [("pointing", "Pointing game"), ("energy_in_tumor", "CAM energy inside tumor")]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    width, x = 0.26, np.arange(len(models))
    for ax, (col, title) in zip(axes, metrics):
        for k, (e, name) in enumerate(exps):
            vals = [tabs[e].loc[m, col] for m in models]
            ax.bar(x + (k - 1) * (width + 0.02), vals, width, color=SERIES[k], label=name)
            if k == 2:
                for xi, v in zip(x + (k - 1) * (width + 0.02), vals):
                    ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=TEXT)
        if col == "energy_in_tumor":
            chance = tabs["cv4_attn"]["chance_energy"].mean()
            ax.axhline(chance, color=TEXT_2, lw=0.8)
            # the line is the chance level (tumor area fraction); explained in the report caption
        ax.set_title(title, fontsize=9, color=TEXT)
        ax.set_xticks(x, [LABELS[m] for m in models], fontsize=8)
        ax.set_ylim(0, 1.0 if col == "pointing" else 0.55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8)
    fig.savefig(out / "gradcam_metrics.pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("report/figures"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_accuracy(args.results, args.out)
    fig_progression(args.results, args.out)
    fig_gradcam(args.results, args.out)
    for exp in ("cv4_improved", "cv4_attn"):
        src = args.results / "gradcam" / exp / "figures" / f"{exp}_fold1_EfficientNet_B3.png"
        shutil.copy(src, args.out / f"cam_{exp}_effnet.png")
    print(sorted(p.name for p in args.out.iterdir()))


if __name__ == "__main__":
    main()
