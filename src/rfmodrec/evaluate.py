"""Metrics and plots: accuracy per SNR, confusion matrix, inference speed."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, ys, snrs = [], [], []
    for x, y, snr in loader:
        preds.append(model(x.to(device)).argmax(1).cpu().numpy())
        ys.append(y.numpy())
        snrs.append(snr.numpy())
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(snrs)


def accuracy_by_snr(pred, y, snr) -> dict[int, float]:
    return {int(s): float((pred[snr == s] == y[snr == s]).mean()) for s in np.unique(snr)}


@torch.no_grad()
def inference_ms_per_frame(model, length: int, device, batch: int = 256, reps: int = 10):
    model.eval()
    x = torch.randn(batch, 2, length, device=device)
    model(x)
    t0 = time.perf_counter()
    for _ in range(reps):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (reps * batch) * 1e3


def save_report(out_dir: Path, classes: list[str], pred, y, snr, extra: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    by_snr = accuracy_by_snr(pred, y, snr)
    high = snr >= 10
    metrics = {
        "accuracy": float((pred == y).mean()),
        "accuracy_snr_ge_10": float((pred[high] == y[high]).mean()) if high.any() else None,
        "accuracy_by_snr": by_snr,
        **extra,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(by_snr), list(by_snr.values()), marker="o")
    ax.set(xlabel="SNR (dB)", ylabel="Accuracy", ylim=(0, 1), title="Accuracy vs SNR")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_snr.png", dpi=150)
    plt.close(fig)

    if high.any():
        cm = confusion_matrix(y[high], pred[high], labels=range(len(classes)), normalize="true")
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
        ax.set_yticks(range(len(classes)), classes, fontsize=7)
        ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix, SNR >= 10 dB")
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_snr_ge_10.png", dpi=150)
        plt.close(fig)
    return metrics
