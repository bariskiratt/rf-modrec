import h5py
import numpy as np
import pytest
import torch

from rfmodrec import data, models, synthetic, train


@pytest.mark.parametrize("name", sorted(models.MODELS))
def test_model_shapes(name):
    m = models.build(name, num_classes=24)
    assert m(torch.randn(4, 2, 1024)).shape == (4, 24)


def test_synthetic_split_is_stratified():
    s = synthetic.generate(per_class_snr=20, snrs=(0, 10))
    tr, va, te = data.stratified_split(s)
    assert len(set(tr) | set(va) | set(te)) == len(s.y)
    assert set(s.y[te]) == set(range(len(synthetic.SYNTH_CLASSES)))


def test_prepare_subset_roundtrip(tmp_path):
    # Tiny file with the real layout: blocks of 4096 frames per (class, SNR).
    classes, snrs, block = 2, [-20, 0], 4096
    n = classes * len(snrs) * block
    with h5py.File(tmp_path / "rml.hdf5", "w") as f:
        f["X"] = np.random.randn(n, 1024, 2).astype(np.float32)
        y = np.repeat(np.eye(24, dtype=np.int64)[:classes], len(snrs) * block, axis=0)
        f["Y"] = y
        f["Z"] = np.tile(np.repeat(snrs, block), classes)[:, None]
    data.prepare_subset(tmp_path / "rml.hdf5", tmp_path / "out", min_snr=-10, per_class_snr=8)
    s = data.load_prepared(tmp_path / "out")
    assert s.x.shape == (16, 2, 1024)
    assert set(s.snr) == {0}
    assert sorted(set(s.y)) == [0, 1]


def test_train_runs_and_learns(tmp_path):
    m = train.main(["--synthetic", "--model", "cnn", "--epochs", "5",
                    "--workers", "0", "--out", str(tmp_path)])
    # 8 classes -> chance is 0.125; a few epochs on tiny data should clear 2x chance.
    assert m["accuracy_by_snr"][20] > 0.25
    assert (tmp_path / "accuracy_vs_snr.png").exists()


def test_dataset_pickles_memmap_as_path(tmp_path):
    # Spawned DataLoader workers (macOS) unpickle the dataset; it must not carry the array.
    import pickle

    np.save(tmp_path / "x.npy", np.random.randn(64, 2, 1024).astype(np.float32))
    np.save(tmp_path / "y.npy", np.zeros(64, dtype=np.int64))
    np.save(tmp_path / "snr.npy", np.zeros(64, dtype=np.int64))
    ds = data.IQDataset(data.load_prepared(tmp_path), np.arange(64))
    blob = pickle.dumps(ds)
    assert len(blob) < 64 * 2 * 1024 * 4 / 4
    back = pickle.loads(blob)
    assert np.array_equal(back.split.x[3], ds.split.x[3])
