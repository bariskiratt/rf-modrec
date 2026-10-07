"""Small synthetic I/Q generator for tests and for running the pipeline before
the real dataset is downloaded. Not a substitute for RadioML: no multipath,
no real hardware impairments, only 8 linear digital modulations.
"""

from __future__ import annotations

import numpy as np

from .data import Split

SYNTH_CLASSES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "4ASK", "OOK", "16APSK"]


def _constellation(name: str) -> np.ndarray:
    if name == "BPSK":
        c = np.array([1, -1], dtype=complex)
    elif name.endswith("PSK") and name[0].isdigit() and "A" not in name:
        m = int(name[:-3])
        c = np.exp(2j * np.pi * np.arange(m) / m)
    elif name == "QPSK":
        c = np.exp(2j * np.pi * (np.arange(4) + 0.5) / 4)
    elif name.endswith("QAM"):
        m = int(np.sqrt(int(name[:-3])))
        g = np.arange(m) - (m - 1) / 2
        c = (g[:, None] + 1j * g[None, :]).ravel()
    elif name == "4ASK":
        c = np.array([-3, -1, 1, 3], dtype=complex)
    elif name == "OOK":
        c = np.array([0, 1], dtype=complex)
    elif name == "16APSK":
        c = np.concatenate(
            [np.exp(2j * np.pi * (np.arange(4) + 0.5) / 4),
             2.7 * np.exp(2j * np.pi * (np.arange(12) + 0.5) / 12)]
        )
    else:
        raise ValueError(name)
    return c / np.sqrt(np.mean(np.abs(c) ** 2))


def _rrc(beta: float, sps: int, span: int) -> np.ndarray:
    t = np.arange(-span * sps, span * sps + 1) / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif beta and abs(abs(4 * beta * ti) - 1) < 1e-8:
            h[i] = beta / np.sqrt(2) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            h[i] = (
                np.sin(np.pi * ti * (1 - beta))
                + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            ) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h**2))


def generate(
    per_class_snr: int = 64,
    snrs: tuple[int, ...] = (-10, -4, 0, 4, 10, 20),
    length: int = 1024,
    sps: int = 8,
    seed: int = 0,
) -> Split:
    rng = np.random.default_rng(seed)
    h = _rrc(0.35, sps, 6)
    n_sym = length // sps + len(h) // sps + 2
    xs, ys, zs = [], [], []
    for label, name in enumerate(SYNTH_CLASSES):
        c = _constellation(name)
        for snr in snrs:
            sym = rng.choice(c, size=(per_class_snr, n_sym))
            up = np.zeros((per_class_snr, n_sym * sps), dtype=complex)
            up[:, ::sps] = sym
            sig = np.stack([np.convolve(u, h, mode="same") for u in up])
            start = rng.integers(0, sig.shape[1] - length, size=per_class_snr)
            sig = np.stack([s[o : o + length] for s, o in zip(sig, start)])
            # Carrier phase and small frequency offset.
            n = np.arange(length)
            phase = rng.uniform(0, 2 * np.pi, (per_class_snr, 1))
            df = rng.uniform(-1e-3, 1e-3, (per_class_snr, 1))
            sig = sig * np.exp(1j * (phase + 2 * np.pi * df * n))
            p = np.mean(np.abs(sig) ** 2, axis=1, keepdims=True)
            noise_p = p / 10 ** (snr / 10)
            sig = sig + np.sqrt(noise_p / 2) * (
                rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape)
            )
            xs.append(np.stack([sig.real, sig.imag], axis=1).astype(np.float32))
            ys.append(np.full(per_class_snr, label))
            zs.append(np.full(per_class_snr, snr))
    return Split(x=np.concatenate(xs), y=np.concatenate(ys), snr=np.concatenate(zs))
