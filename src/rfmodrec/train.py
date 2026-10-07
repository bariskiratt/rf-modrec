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


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.synthetic:
        split, classes = synthetic.generate(seed=args.seed), synthetic.SYNTH_CLASSES
    else:
        split, classes = data.load_prepared(args.data), data.RADIOML_2018_CLASSES
    tr, va, te = data.stratified_split(split, seed=args.seed)
    loader = lambda idx, shuffle: DataLoader(  # noqa: E731
        data.IQDataset(split, idx),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
    )
    train_dl, val_dl, test_dl = loader(tr, True), loader(va, False), loader(te, False)

    length = split.x.shape[-1]
    model = models.build(args.model, len(classes), length).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    out = args.out or Path("runs") / f"{args.model}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.model}: {n_params:,} params, {len(tr)}/{len(va)}/{len(te)} frames, {device}")

    best, stale, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        pred, y, _ = evaluate.predict(model, val_dl, device)
        val_acc = float((pred == y).mean())
        history.append({"epoch": epoch, "train_loss": total / len(tr), "val_acc": val_acc})
        print(f"epoch {epoch:3d}  loss {total / len(tr):.4f}  val_acc {val_acc:.4f}  {time.time() - t0:.0f}s")
        if val_acc > best:
            best, stale = val_acc, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print("early stop")
                break

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
