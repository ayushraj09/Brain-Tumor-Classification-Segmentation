"""Train / evaluate classifiers on the official patient-wise 5-fold split (test = fold k, validation = fold k+1).

Examples
--------
Improved recipe, 3 tumor classes (the literature setting)::

    python scripts/train_classifier.py --exp cv3_improved --recipe improved --split patient --classes 3

Grad-CAM-guided attention training (penalises CAM energy outside the dilated tumor mask)::

    python scripts/train_classifier.py --exp cv4_attn --recipe improved --split patient --attn-weight 1.0
"""

from __future__ import annotations

import argparse
import json
import time

import torch.nn.functional as F

from common import *  # noqa: F403


def attention_loss(feats: torch.Tensor, logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, dilate: int,
                   form: str = "one_minus_inside"):
    """Differentiable Grad-CAM of the true class; returns 1 − (fraction of CAM energy inside the dilated tumor
    mask), averaged over samples that have a mask. Dilation keeps the peritumoral context, which Cheng et al.
    (2015) showed is informative for tumor-type classification. Writing the loss as 1 − inside (rather than
    outside / total) makes an all-zero CAM the *worst* case, so the network cannot satisfy it by collapsing its
    Grad-CAM to zero. ``form="outside_ratio"`` is the first formulation (outside / total); it is identical whenever
    the CAM has positive mass but scores an all-zero CAM as perfect, which AlexNet exploited."""
    has_mask = mask.flatten(1).any(1)
    if not has_mask.any():
        return logits.new_zeros(())
    score = logits.gather(1, y[:, None]).sum()
    grads = torch.autograd.grad(score, feats, create_graph=True)[0]
    cam = F.relu((grads.mean((2, 3), keepdim=True) * feats).sum(1, keepdim=True))[has_mask].float()
    cam = F.interpolate(cam, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    m = F.max_pool2d(mask[has_mask, None].float(), dilate, stride=1, padding=dilate // 2)
    total = cam.flatten(1).sum(1) + 1e-6
    if form == "outside_ratio":
        return ((cam * (1 - m)).flatten(1).sum(1) / total).mean()
    return (1 - (cam * m).flatten(1).sum(1) / total).mean()


def run_one(model_name: str, df: pd.DataFrame, split: tuple, recipe: Recipe, class_names: list[str], tag: str,
            seed: int, attn_dilate: int, boxes: dict | None = None, roi_context: float = 0.0) -> dict:
    set_seed(seed)
    size = recipe.size_for(model_name)
    train_idx, val_idx, test_idx = split
    use_mask = recipe.attn_weight > 0
    if recipe.aug == "none" and recipe.name == "baseline":
        train_tf = eval_tf = torchvision_eval_transform(size)
    else:
        train_tf = {"none": albu_eval, "light": light_train_aug}[recipe.aug](size)
        eval_tf = albu_eval(size)
    roi = dict(boxes=boxes, context=roi_context) if roi_context > 0 else {}
    train_loader = make_loader(MRIDataset(df, train_idx, train_tf, size, with_mask=use_mask, roi="gt" if roi else None, **roi),
                               recipe.batch_size, True, seed)
    val_loader = make_loader(MRIDataset(df, val_idx, eval_tf, size, roi="pred" if roi else None, **roi),
                             recipe.batch_size, False) if val_idx else None
    test_loader = make_loader(MRIDataset(df, test_idx, eval_tf, size, roi="pred" if roi else None, **roi), recipe.batch_size, False)

    model = build_model(model_name, len(class_names), size).to(DEVICE)
    feats = {}
    if use_mask:
        for mod in model.modules():
            if hasattr(mod, "inplace"):
                mod.inplace = False
        gradcam_layer(model, model_name).register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))

    ema = None
    if recipe.ema > 0:
        ema = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(recipe.ema), use_buffers=True)
    eval_model = ema.module if ema is not None else model

    criterion = nn.CrossEntropyLoss(label_smoothing=recipe.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(recipe, len(train_loader)))

    ckpt = CKPT_DIR / tag / f"{model_name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    history, best, best_state, best_epoch = [], -1.0, None, 0
    for epoch in range(1, recipe.epochs + 1):
        model.train()
        t0, tot_loss, tot_attn, correct, n = time.time(), 0.0, 0.0, 0, 0
        for batch in train_loader:
            x, y = batch[0].to(DEVICE, non_blocking=True), batch[1].to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=recipe.amp):
                logits, aux = split_logits(model(x))
                loss = criterion(logits.float(), y)
                if aux is not None and recipe.aux_loss > 0:
                    loss = loss + recipe.aux_loss * criterion(aux.float(), y)
                if use_mask:
                    attn = attention_loss(feats["a"], logits.float(), y, batch[2].to(DEVICE), attn_dilate,
                                          recipe.attn_form)
                    loss = loss + recipe.attn_weight * attn
                    tot_attn += attn.item() * y.size(0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update_parameters(model)
            tot_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        if val_loader is None:  # refit: no validation set, keep the final (end-of-schedule) weights
            vm, score = {"accuracy": float("nan"), "macro_f1": float("nan")}, float(epoch)
        else:
            yv, pv = predict(eval_model, val_loader, amp=recipe.amp)
            vm = metrics(yv, pv, class_names)
            score = vm["accuracy"] if recipe.select_metric == "acc" else vm["macro_f1"]
        history.append(dict(epoch=epoch, train_loss=tot_loss / n, train_acc=correct / n, train_attn=tot_attn / n,
                            val_acc=vm["accuracy"], val_macro_f1=vm["macro_f1"], lr=optimizer.param_groups[0]["lr"],
                            sec=time.time() - t0))
        if score > best:
            best, best_epoch, best_state = score, epoch, deepcopy(eval_model.state_dict())
            torch.save(best_state, ckpt)
        print(f"[{tag}/{model_name}] ep {epoch:02d} loss={tot_loss / n:.4f} acc={correct / n:.4f} "
              f"attn={tot_attn / n:.3f} | val acc={vm['accuracy']:.4f} mF1={vm['macro_f1']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    eval_model.load_state_dict(best_state)
    # validation probabilities of the selected checkpoint (same TTA as test) — used to choose ensembles without
    # touching the test fold
    yv, pv = predict(eval_model, val_loader, tta=recipe.tta, amp=recipe.amp) if val_loader else ([], [])
    yt, pt = predict(eval_model, test_loader, tta=recipe.tta, amp=recipe.amp)
    tm = metrics(yt, pt, class_names)
    print(f"[{tag}/{model_name}] best epoch {best_epoch} | TEST acc={tm['accuracy']:.4f} "
          f"wF1={tm['f1']:.4f} mF1={tm['macro_f1']:.4f}", flush=True)
    return dict(model=model_name, tag=tag, recipe=asdict(recipe), classes=class_names, best_epoch=best_epoch,
                best_val=best, test=tm, history=history, test_idx=list(test_idx), test_paths=df.path.iloc[test_idx].tolist(),
                test_true=yt, test_probs=pt, val_true=yv, val_probs=pv, seed=seed, refit=not val_idx, roi_context=roi_context, n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--recipe", default="baseline", choices=list(RECIPES))
    p.add_argument("--split", default="patient", choices=["patient"], help="kept for explicit command lines")
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--classes", type=int, default=4, choices=[3, 4])
    p.add_argument("--models", nargs="+", default=MODEL_NAMES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int)
    p.add_argument("--attn-weight", type=float, default=0.0)
    p.add_argument("--attn-dilate", type=int, default=31, help="mask dilation kernel (pixels at input resolution)")
    p.add_argument("--image-size", type=int, help="input size of every model except MLP / AlexNet (default: recipe)")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off)")
    p.add_argument("--refit", action="store_true",
                   help="train on the 4 non-test folds (no validation fold) for the full schedule "
                        "and keep the final weights; configurations must be chosen beforehand on validation runs")
    p.add_argument("--roi-context", type=float, default=0.0,
                   help=">0: classify tumor-region crops (side = context × box side); val/test boxes from "
                        "results/boxes/fold<k>.json (scripts/predict_boxes.py), training boxes from the GT mask")
    p.add_argument("--attn-form", default="one_minus_inside", choices=["one_minus_inside", "outside_ratio"])
    args = p.parse_args()

    recipe = deepcopy(RECIPES[args.recipe])
    recipe.attn_weight = args.attn_weight
    recipe.attn_form = args.attn_form
    if args.epochs:
        recipe.epochs = args.epochs
    recipe.ema = args.ema
    if args.image_size:
        recipe.image_size = {m: args.image_size for m in args.models if m not in ("MLP", "AlexNet")}
    class_names = CLASS_NAMES if args.classes == 4 else TUMOR_CLASSES
    df = load_samples(class_names)
    splits = {k: patient_cv_split(df, k) for k in args.folds}
    if args.refit:
        splits = {k: (tr + va, [], te) for k, (tr, va, te) in splits.items()}

    for fold, split in splits.items():
        tag = f"{args.exp}/fold{fold}"
        for name in args.models:
            out = RESULTS_DIR / tag / f"{name}.json"
            if out.exists():
                print(f"skip {out}")
                continue
            boxes = None
            if args.roi_context > 0:
                boxes = json.loads((RESULTS_DIR / "boxes" / f"fold{fold}.json").read_text())
            save_json(run_one(name, df, split, recipe, class_names, tag, args.seed, args.attn_dilate, boxes,
                              args.roi_context), out)


if __name__ == "__main__":
    main()
