"""Validation-only model and ensemble selection for patient-level CV experiments.

Every candidate is a (experiment, model) pair trained on all 5 folds whose JSON files contain ``val_probs``
(written by train_classifier.py). For fold k the validation set is fold k+1, so the 5 validation sets together
cover every image once. Candidates are ranked by pooled validation accuracy; an ensemble is grown greedily
(Caruana et al., 2004: add the member that most increases pooled validation accuracy, with replacement, until no
member helps). Test accuracy is computed only for the reported single models and the final ensemble.

    python scripts/select_ensemble.py --exps p3_convnext384_ema p3_effv2s384_ema p3_effb3_ema

``--groups`` instead selects at the configuration level: each group (``name=exp1,exp2,...``, typically one
configuration trained with several seeds) contributes all its members, and every subset of groups is scored by pooled
validation accuracy of the uniform average. This is less prone to fitting validation noise than per-model greedy
selection.

    python scripts/select_ensemble.py --groups effb3=p3_full_effb3,p3_full_effb3_s1 convnext=p3_convnext384_ema
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from common import RESULTS_DIR, save_json


def load(exps: list[str], need_val: bool = True) -> dict[str, dict[int, dict]]:
    cands: dict[str, dict[int, dict]] = {}
    for exp in exps:
        for p in sorted((RESULTS_DIR / exp).glob("fold*/*.json")):
            r = json.loads(p.read_text())
            if need_val and not r.get("val_probs"):
                continue
            cands.setdefault(f"{exp}:{r['model']}", {})[int(p.parent.name[4:])] = r
    return {k: v for k, v in cands.items() if len(v) == 5}


def fold_acc(members: list[str], cands, split: str) -> list[float]:
    accs = []
    for k in range(1, 6):
        y = np.array(cands[members[0]][k][f"{split}_true"])
        p = np.mean([np.array(cands[m][k][f"{split}_probs"]) for m in members], axis=0)
        accs.append(float((p.argmax(1) == y).mean()))
    return accs


def pooled(members: list[str], cands, split: str) -> float:
    correct = total = 0
    for k in range(1, 6):
        y = np.array(cands[members[0]][k][f"{split}_true"])
        p = np.mean([np.array(cands[m][k][f"{split}_probs"]) for m in members], axis=0)
        correct += int((p.argmax(1) == y).sum())
        total += len(y)
    return correct / total


def score(args) -> None:
    """Test metrics of a fixed (pre-registered) uniform ensemble, e.g. refit models that have no validation set."""
    from common import TUMOR_CLASSES, metrics

    cands = load(args.score, need_val=False)
    ms = list(cands)
    t = fold_acc(ms, cands, "test")
    y = np.concatenate([np.array(cands[ms[0]][k]["test_true"]) for k in range(1, 6)])
    p = np.concatenate([np.mean([np.array(cands[m][k]["test_probs"]) for m in ms], axis=0) for k in range(1, 6)])
    m = metrics(y, p, TUMOR_CLASSES if p.shape[1] == 3 else None)
    print(f"{len(ms)} models: {ms}")
    print(f"test accuracy {100 * np.mean(t):.2f} ± {100 * np.std(t, ddof=1):.2f} | pooled {100 * m['accuracy']:.2f} | "
          f"macro-F1 {100 * m['macro_f1']:.2f} | per fold {[round(100 * a, 2) for a in t]}")
    print("recall:", {c: round(v["recall"], 3) for c, v in m["per_class"].items()})
    save_json({"members": ms, "test_folds": t, "test_pooled": m}, RESULTS_DIR / args.out)


def select_groups(args) -> None:
    groups = {g.split("=")[0]: g.split("=")[1].split(",") for g in args.groups}
    cands = load([e for v in groups.values() for e in v])
    members = {g: [k for k in cands if k.split(":")[0] in exps] for g, exps in groups.items()}
    rows = []
    for r in range(1, len(groups) + 1):
        for combo in itertools.combinations(groups, r):
            ms = [m for g in combo for m in members[g]]
            rows.append((pooled(ms, cands, "val"), combo, ms))
    rows.sort(key=lambda t: t[0], reverse=True)
    print("configuration subsets ranked by pooled validation accuracy:")
    for v, combo, _ in rows[:10]:
        print(f"  val {100 * v:.2f}  {' + '.join(combo)}")
    v, combo, ms = rows[0]
    t = fold_acc(ms, cands, "test")
    print(f"\nselected (validation only): {' + '.join(combo)}  ({len(ms)} models, val {100 * v:.2f})")
    print(f"test accuracy: {100 * np.mean(t):.2f} ± {100 * np.std(t, ddof=1):.2f}  pooled {100 * pooled(ms, cands, 'test'):.2f}"
          f"  per fold {[round(100 * a, 2) for a in t]}")
    save_json({"ranking": [{"groups": c, "val_pooled": vv} for vv, c, _ in rows], "selected": combo, "members": ms,
               "val_pooled": v, "test_folds": t, "test_pooled": pooled(ms, cands, "test")}, RESULTS_DIR / args.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=[])
    ap.add_argument("--groups", nargs="+", default=[], help="name=exp1,exp2,... (configuration-level selection)")
    ap.add_argument("--score", nargs="+", default=[], help="experiments forming a fixed uniform ensemble to evaluate")
    ap.add_argument("--max-members", type=int, default=8)
    ap.add_argument("--out", default="selection/ensemble_selection.json")
    args = ap.parse_args()
    if args.score:
        score(args)
        return
    if args.groups:
        select_groups(args)
        return
    cands = load(args.exps)
    rows = sorted(((pooled([c], cands, "val"), c) for c in cands), reverse=True)
    print(f"{'candidate':45s} {'val (pooled)':>12s} {'test mean ± std':>18s}")
    for v, c in rows:
        t = fold_acc([c], cands, "test")
        print(f"{c:45s} {100 * v:12.2f} {100 * np.mean(t):10.2f} ± {100 * np.std(t, ddof=1):.2f}")

    ens, best = [rows[0][1]], rows[0][0]
    while len(ens) < args.max_members:
        trial = max(((pooled(ens + [c], cands, "val"), c) for c in cands), key=lambda t: t[0])
        if trial[0] <= best:
            break
        best, ens = trial[0], ens + [trial[1]]
    t = fold_acc(ens, cands, "test")
    print(f"\nselected ensemble (val {100 * best:.2f}): {ens}")
    print(f"test accuracy: {100 * np.mean(t):.2f} ± {100 * np.std(t, ddof=1):.2f}  per fold {[round(100 * a, 2) for a in t]}"
          f"  pooled {100 * pooled(ens, cands, 'test'):.2f}")
    save_json({"candidates": [{"name": c, "val_pooled": v, "test_folds": fold_acc([c], cands, "test")} for v, c in rows],
               "ensemble": ens, "ensemble_val_pooled": best, "ensemble_test_folds": t,
               "ensemble_test_pooled": pooled(ens, cands, "test")}, RESULTS_DIR / args.out)


if __name__ == "__main__":
    main()
