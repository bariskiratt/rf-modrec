"""Turn the 21 GB RadioML 2018.01A HDF5 file into a smaller, balanced .npy subset.

    python -m rfmodrec.prepare path/to/GOLD_XYZ_OSC.0001_1024.hdf5 data/radioml_subset \
        --per-class-snr 1024

1024 frames per (class, SNR) gives 24 * 26 * 1024 = 638,976 frames, about 5.2 GB.
Start with --per-class-snr 256 (about 1.3 GB) while developing.
"""

import argparse

from .data import prepare_subset

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("hdf5")
    p.add_argument("out_dir")
    p.add_argument("--min-snr", type=int, default=-20)
    p.add_argument("--per-class-snr", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    prepare_subset(a.hdf5, a.out_dir, a.min_snr, a.per_class_snr, a.seed)
