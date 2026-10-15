"""Data loading for RadioML 2018.01A and a synthetic stand-in.

RadioML 2018.01A ships as one HDF5 file (GOLD_XYZ_OSC.0001_1024.hdf5):
    X: (2_555_904, 1024, 2) float32  I/Q samples
    Y: (2_555_904, 24)      int64    one-hot modulation label
    Z: (2_555_904, 1)       int64    SNR in dB, -20..30 step 2

Random access into that file is slow, so `prepare_subset` copies a filtered,
class/SNR-balanced subset into .npy files once; training then memory-maps them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# The classes.txt that ships with the dataset is known to be in the wrong order.
# This is the order the community converged on; confirm it in week 1 by plotting
# high-SNR constellations per label (see README).
RADIOML_2018_CLASSES = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]


@dataclass
class Split:
    x: np.ndarray  # (N, 2, L) float32, channels-first for Conv1d
    y: np.ndarray  # (N,) int64
    snr: np.ndarray  # (N,) int64


def prepare_subset(
    hdf5_path: str | Path,
    out_dir: str | Path,
    min_snr: int = -20,
    per_class_snr: int = 1024,
    seed: int = 0,
) -> None:
    """Copy a balanced subset of the HDF5 file into x.npy / y.npy / snr.npy.

    The file stores 4096 consecutive frames per (class, SNR) pair, so we sample
    `per_class_snr` indices from each block instead of scanning labels.
    """
    import h5py

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "r") as f:
        n = f["Z"].shape[0]
        block = 4096
        picks = []
        for start in range(0, n, block):
            snr = int(f["Z"][start, 0])
            if snr != int(f["Z"][start + block - 1, 0]) or not np.array_equal(
                f["Y"][start], f["Y"][start + block - 1]
            ):
                raise ValueError(f"block at {start} is not one (class, SNR) pair")
            if snr < min_snr:
                continue
            idx = rng.choice(block, size=min(per_class_snr, block), replace=False)
            picks.append(np.sort(idx) + start)
        idx = np.concatenate(picks)
        x = np.empty((len(idx), 2, 1024), dtype=np.float32)
        for i in range(0, len(idx), 65536):
            chunk = idx[i : i + 65536]
            x[i : i + len(chunk)] = f["X"][chunk].transpose(0, 2, 1)
        y = f["Y"][idx].argmax(axis=1).astype(np.int64)
        snr = f["Z"][idx, 0].astype(np.int64)
    np.save(out_dir / "x.npy", x)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "snr.npy", snr)


def load_prepared(data_dir: str | Path) -> Split:
    d = Path(data_dir)
    return Split(
        x=np.load(d / "x.npy", mmap_mode="r"),
        y=np.load(d / "y.npy"),
        snr=np.load(d / "snr.npy"),
    )


def stratified_split(
    split: Split, val: float = 0.1, test: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index arrays (train, val, test) into `split`, stratified on (class, SNR).

    Every (class, SNR) pair should land in all three parts in the same proportions,
    otherwise accuracy-vs-SNR curves compare different class mixes. The three arrays
    must be disjoint and together cover every index.
    """
    raise NotImplementedError("week 2-3")


class IQDataset(Dataset):
    """Frames `split.x[indices]` as (x, label, snr) tuples for a DataLoader.

    x is a float32 tensor of shape (2, L). With `normalize`, scale each frame to unit
    average power so the model cannot use the receive gain as a shortcut.
    """

    def __init__(self, split: Split, indices: np.ndarray, normalize: bool = True):
        self.split = split
        self.indices = indices
        self.normalize = normalize

    def __getstate__(self):
        # DataLoader workers on macOS and Windows are spawned and get a pickled copy of
        # the dataset. A pickled memmap carries the whole array, so send the path instead.
        state = self.__dict__.copy()
        if isinstance(self.split.x, np.memmap):
            state["split"] = replace(self.split, x=str(self.split.x.filename))
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if isinstance(self.split.x, str):
            self.split = replace(self.split, x=np.load(self.split.x, mmap_mode="r"))

    def __len__(self) -> int:
        raise NotImplementedError("week 2-3")

    def __getitem__(self, i: int):
        raise NotImplementedError("week 2-3")
