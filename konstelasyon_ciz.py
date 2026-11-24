"""Her etiket için yüksek SNR'da bir çerçevenin takımyıldızını (I-Q) ve genliğini çizer.

Kullanım (repo kökünden):
    python konstelasyon_ciz.py data/radioml_256 --snr 30

Çıktı: data/radioml_256/konstelasyon_snr30.png ve genlik_snr30.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")
from rfmodrec.data import RADIOML_2018_CLASSES  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("data_dir")
p.add_argument("--snr", type=int, default=30)
p.add_argument("--frame", type=int, default=0, help="kaçıncı örnek çerçeve (0, 1, 2...)")
a = p.parse_args()

d = Path(a.data_dir)
x = np.load(d / "x.npy", mmap_mode="r")  # (N, 2, 1024): kanal 0 = I, kanal 1 = Q
y = np.load(d / "y.npy")
snr = np.load(d / "snr.npy")

fig_c, ax_c = plt.subplots(4, 6, figsize=(18, 12))
fig_a, ax_a = plt.subplots(4, 6, figsize=(18, 10))
for label in range(24):
    idx = np.flatnonzero((y == label) & (snr == a.snr))
    i, q = x[idx[a.frame]]
    guc = np.sqrt(np.mean(i**2 + q**2))  # ortalama gücü 1 yap
    i, q = i / guc, q / guc
    title = f"etiket {label}: {RADIOML_2018_CLASSES[label]}"

    ax = ax_c.flat[label]
    ax.scatter(i, q, s=2, alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = ax_a.flat[label]
    ax.plot(np.sqrt(i**2 + q**2)[:256], lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 3)
    ax.set_xticks([])

fig_c.suptitle(f"Takımyıldızlar (I yatay, Q dikey), SNR = {a.snr} dB")
fig_a.suptitle(f"Genlik |x| zamana göre, ilk 256 örnek, SNR = {a.snr} dB")
fig_c.tight_layout()
fig_a.tight_layout()
fig_c.savefig(d / f"konstelasyon_snr{a.snr}.png", dpi=120)
fig_a.savefig(d / f"genlik_snr{a.snr}.png", dpi=120)
print("kaydedildi:", d / f"konstelasyon_snr{a.snr}.png", "ve", d / f"genlik_snr{a.snr}.png")
