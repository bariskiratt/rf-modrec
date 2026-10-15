"""Train and evaluate one model.

    # real data (after `python -m rfmodrec.prepare`)
    python -m rfmodrec.train --data data/radioml_subset --model resnet --epochs 30

    # synthetic data, runs on a laptop CPU in a minute
    python -m rfmodrec.train --synthetic --model cnn --epochs 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import data, evaluate, models, synthetic


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="directory written by rfmodrec.prepare")
    src.add_argument("--synthetic", action="store_true")
    p.add_argument("--model", choices=sorted(models.MODELS), default="cnn")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(split, tr, va, te, args) -> tuple[DataLoader, DataLoader, DataLoader]:
    """DataLoaders over data.IQDataset for the train, val and test indices.

    Only the training loader should shuffle. Use args.batch_size and args.workers.
    """
    raise NotImplementedError("week 3")


def fit(model, train_dl, val_dl, args, device, out: Path) -> tuple[float, list[dict]]:
    """Train `model` and return (best validation accuracy, per-epoch history).

    Each epoch: one pass over train_dl, then validation accuracy on val_dl.
    Save the weights to out / "best.pt" whenever validation accuracy improves, and
    stop early after args.patience epochs without improvement. A history entry
    should look like {"epoch": 3, "train_loss": 1.23, "val_acc": 0.45}.
    """
    raise NotImplementedError("week 3")


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = pick_device()

    if args.synthetic:
        split, classes = synthetic.generate(seed=args.seed), synthetic.SYNTH_CLASSES
    else:
        split, classes = data.load_prepared(args.data), data.RADIOML_2018_CLASSES
    tr, va, te = data.stratified_split(split, seed=args.seed)
    train_dl, val_dl, test_dl = make_loaders(split, tr, va, te, args)

    length = split.x.shape[-1]
    model = models.build(args.model, len(classes), length).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    out = args.out or Path("runs") / f"{args.model}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.model}: {n_params:,} params, {len(tr)}/{len(va)}/{len(te)} frames, {device}")

    best, history = fit(model, train_dl, val_dl, args, device, out)

    (out / "history.json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    pred, y, snr = evaluate.predict(model, test_dl, device)
    metrics = evaluate.save_report(
        out, classes, pred, y, snr,
        extra={
            "model": args.model,
            "params": n_params,
            "best_val_acc": best,
            "ms_per_frame": evaluate.inference_ms_per_frame(model, length, device),
            "device": str(device),
        },
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "accuracy_by_snr"}, indent=2))
    return metrics


if __name__ == "__main__":
    main()
