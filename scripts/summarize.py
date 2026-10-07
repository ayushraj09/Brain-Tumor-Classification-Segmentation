"""Aggregate classification / segmentation results into markdown tables (results/summary.md + summary.json).

Patient-CV experiments report mean ± std over the 5 official folds and the pooled accuracy over all 3064 (3-class)
or 3459 (4-class) test predictions. ``ENS`` averages the softmax outputs of the CNNs of the same fold.
"""

from __future__ import annotations

import argparse
import json

from common import RESULTS_DIR, metrics, np, pd

ENSEMBLE = ["AlexNet", "InceptionV3", "EfficientNet_B3"]
ORDER = ["MLP", "AlexNet", "InceptionV3", "EfficientNet_B3", "ENS"]


def load_runs(exp: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((RESULTS_DIR / exp).glob("**/*.json"))
            if p.stem in ORDER and "gradcam" not in str(p)]


def classification_table(exp: str) -> tuple[pd.DataFrame, dict]:
    runs = load_runs(exp)
    if not runs:
        return pd.DataFrame(), {}
    by_fold: dict[str, dict[str, dict]] = {}
    for r in runs:
        by_fold.setdefault(r["tag"], {})[r["model"]] = r
    class_names = runs[0]["classes"]
    rows, pooled = [], {}
    for tag, models in by_fold.items():
        if all(m in models for m in ENSEMBLE):
            y = np.array(models[ENSEMBLE[0]]["test_true"])
            p = np.mean([np.array(models[m]["test_probs"]) for m in ENSEMBLE], axis=0)
            models["ENS"] = {"model": "ENS", "test": metrics(y, p, class_names), "test_true": y, "test_probs": p}
        for name, r in models.items():
            rows.append(dict(fold=tag, model=name, **{k: r["test"][k] for k in ("accuracy", "precision", "recall", "f1", "macro_f1")}))
            pooled.setdefault(name, ([], []))
            pooled[name][0].append(np.array(r["test_true"]))
            pooled[name][1].append(np.array(r["test_probs"]))
    df = pd.DataFrame(rows)
    agg = df.groupby("model")[["accuracy", "precision", "recall", "f1", "macro_f1"]].agg(["mean", "std"])
    out = pd.DataFrame(index=[m for m in ORDER if m in agg.index])
    nfold = df.fold.nunique()
    for k in ["accuracy", "precision", "recall", "f1", "macro_f1"]:
        out[k] = [f"{agg.loc[m, (k, 'mean')]:.4f}" + (f" ± {agg.loc[m, (k, 'std')]:.4f}" if nfold > 1 else "") for m in out.index]
    pooled_m = {m: metrics(np.concatenate(pooled[m][0]), np.concatenate(pooled[m][1]), class_names) for m in out.index}
    out["pooled_acc"] = [f"{pooled_m[m]['accuracy']:.4f}" for m in out.index]
    out["n_folds"] = nfold
    for c in class_names:
        out[f"recall_{c}"] = [f"{pooled_m[m]['per_class'][c]['recall']:.3f}" for m in out.index]
    return out, pooled_m


def segmentation_table(exp: str) -> pd.DataFrame:
    files = sorted((RESULTS_DIR / exp).glob("**/maskrcnn.json"))
    if not files:
        return pd.DataFrame()
    rows = []
    for f in files:
        r = json.loads(f.read_text())
        for key in ("test_at_0_4", "test_tuned"):
            rows.append(dict(fold=r["exp"], setting=f"thr={r[key]['threshold']}" if key == "test_tuned" else "thr=0.4",
                             key=key, **{k: r[key][k] for k in ("precision", "recall", "specificity", "f1",
                                                                "mean_iou", "mean_dice", "mean_iou_detected")}))
    df = pd.DataFrame(rows)
    cols = ["precision", "recall", "specificity", "f1", "mean_iou", "mean_dice", "mean_iou_detected"]
    agg = df.groupby("key")[cols].agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for k in cols:
        out[k] = [f"{agg.loc[i, (k, 'mean')]:.4f}" + (f" ± {agg.loc[i, (k, 'std')]:.4f}" if len(files) > 1 else "") for i in agg.index]
    out["thresholds"] = df.groupby("key").setting.apply(lambda s: ",".join(sorted(set(s))))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cls", nargs="*", default=[])
    p.add_argument("--seg", nargs="*", default=[])
    args = p.parse_args()
    lines, js = [], {}
    for exp in args.cls:
        t, pooled = classification_table(exp)
        if t.empty:
            continue
        lines += [f"### {exp}", "", t.to_markdown(), ""]
        js[exp] = {m: {k: v for k, v in pm.items() if k != "per_class"} | {"per_class": pm["per_class"]} for m, pm in pooled.items()}
        print(f"\n### {exp}\n{t.iloc[:, :7].to_string()}")
    for exp in args.seg:
        t = segmentation_table(exp)
        if t.empty:
            continue
        lines += [f"### {exp}", "", t.to_markdown(), ""]
        print(f"\n### {exp}\n{t.to_string()}")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(js, indent=1))


if __name__ == "__main__":
    main()
