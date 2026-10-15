# rf-modrec: RF Modulation Recognition on RadioML 2018

Classify the modulation of a raw I/Q radio frame (24 classes, -20 to +30 dB SNR) and
compare three model families on accuracy, low-SNR robustness, and inference cost.

> Status: project skeleton. The models, dataset, split and training loop are stubs I am
> writing myself (see PLAN.md); `pytest` is their acceptance test. Results tables below
> are filled in as experiments run.

## Why this problem

Recognising an unknown emitter's modulation is a core step in spectrum monitoring,
cognitive radio, and electronic support systems. The hard part is low SNR, where
classes such as 16QAM/64QAM or AM-DSB variants become nearly indistinguishable.

## Questions this project answers

1. How do a VGG-style CNN, a residual network, and a Transformer compare at equal
   training budget?
2. Where does each model break down as SNR drops, and which classes get confused?
3. How much accuracy is lost when the best model is quantised for fast, small-footprint
   inference?

## Repository layout

```
src/rfmodrec/
  data.py        RadioML loading, balanced subset extraction, stratified split
  synthetic.py   small synthetic generator (tests, and work before the download)
  models.py      cnn / resnet / transformer
  train.py       training loop with early stopping, writes runs/<name>/
  evaluate.py    accuracy vs SNR, confusion matrix, ms per frame
  prepare.py     CLI: HDF5 -> balanced .npy subset
tests/           smoke tests (pytest)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                              # ~30 s on CPU

# pipeline check on synthetic signals (no download needed)
PYTHONPATH=src python -m rfmodrec.train --synthetic --model cnn --epochs 10

# real data: download RadioML 2018.01A from DeepSig, then
PYTHONPATH=src python -m rfmodrec.prepare GOLD_XYZ_OSC.0001_1024.hdf5 data/radioml_subset --per-class-snr 256
PYTHONPATH=src python -m rfmodrec.train --data data/radioml_subset --model resnet --epochs 30
```

Each run writes `metrics.json`, `history.json`, `best.pt`, `accuracy_vs_snr.png` and
`confusion_snr_ge_10.png` to `runs/<model>-<timestamp>/`.

## Data notes

- RadioML 2018.01A: 2,555,904 frames of 1024 complex samples, 4096 frames per
  (class, SNR) pair. The file is about 21 GB.
- The `classes.txt` shipped with the dataset is known to list labels in the wrong order.
  `data.RADIOML_2018_CLASSES` uses the order the community settled on; verify it by
  plotting high-SNR constellations per label before trusting any confusion matrix.
- Splits are stratified on (class, SNR) so every SNR level is equally represented in
  train, validation and test.

## Results

| Model | Params | Acc (all SNR) | Acc (SNR >= 10 dB) | ms / frame (CPU) |
| --- | --- | --- | --- | --- |
| CNN | | | | |
| ResNet | | | | |
| Transformer | | | | |
| ResNet, int8 | | | | |

## References

- T. O'Shea, T. Roy, T. C. Clancy, "Over-the-Air Deep Learning Based Radio Signal
  Classification," IEEE JSTSP, 2018.
- T. O'Shea, J. Corgan, T. C. Clancy, "Convolutional Radio Modulation Recognition
  Networks," EANN, 2016.
