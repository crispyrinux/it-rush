import json
from pathlib import Path

import numpy as np
import pytest

from bisindo.dataset import save_landmarks_npz, save_raw_hands_npz
from bisindo.features import FEATURE_DIM, FEATURE_DIM_V2, NUM_LANDMARKS
from bisindo.model import convert_tflite_mlp_to_keras
from bisindo.train import augment_training_split, run_finetuning, run_training, run_v2_training


def test_augment_training_split_increases_dataset_size():
    rng = np.random.default_rng(0)
    X = rng.random((20, FEATURE_DIM), dtype=np.float32)
    y = np.repeat([0, 1], 10)

    X_aug, y_aug = augment_training_split(
        X,
        y,
        flip_swap=True,
        single_hand_ratio=0.2,
        jitter_std=0.01,
        jitter_copies=1,
        seed=42,
    )

    # 20 original + 20 flip_swap + 4 single_hand + 20 jitter = 64
    assert len(X_aug) == 64
    assert len(y_aug) == 64


def test_run_training_end_to_end_synthetic(tmp_path):
    classes = ["A", "B", "NOTHING"]
    rng = np.random.default_rng(1)

    X_parts = []
    y_parts = []
    for idx, cls in enumerate(classes):
        cluster = rng.normal(loc=0.2 + idx * 0.3, scale=0.03, size=(15, FEATURE_DIM))
        X_parts.append(np.clip(cluster, 0.0, 1.0).astype(np.float32))
        y_parts.append(np.full(15, cls))

    npz_path = tmp_path / "synth.npz"
    save_landmarks_npz(npz_path, np.concatenate(X_parts), np.concatenate(y_parts))

    tflite_out = tmp_path / "artifacts" / "model.tflite"
    labels_out = tmp_path / "artifacts" / "labels.json"
    tfjs_dir = tmp_path / "artifacts" / "tfjs"

    res = run_training(
        npz_paths=[npz_path],
        classes=classes,
        tflite_out=tflite_out,
        labels_out=labels_out,
        tfjs_dir=tfjs_dir,
        epochs=5,
        batch_size=8,
        val_split=0.2,
        test_split=0.2,
        jitter_copies=1,
    )

    assert res.tflite_path.exists()
    assert res.labels_path.exists()
    assert res.tfjs_model_path is not None and res.tfjs_model_path.exists()
    assert (tfjs_dir / "group1-shard1of1.bin").exists()
    assert json.loads((tfjs_dir / "labels.json").read_text(encoding="utf-8")) == classes
    assert json.loads(res.labels_path.read_text(encoding="utf-8")) == classes
    assert 0.0 <= res.test_accuracy <= 1.0


def test_run_finetuning_small_subset_preserves_absent_classes_and_exports(tmp_path):
    base_tflite = Path("artifacts/bisindo_alphabet_model_f32.tflite")
    base_labels = Path("artifacts/bisindo_alphabet_labels.json")
    if not base_tflite.exists() or not base_labels.exists():
        pytest.skip("Pretrained KawanIsyarat TFLite artifacts not present")

    labels_list = json.loads(base_labels.read_text(encoding="utf-8"))
    z_idx = labels_list.index("Z")
    base_keras = convert_tflite_mlp_to_keras(base_tflite)
    w_base, b_base = base_keras.layers[-1].get_weights()

    # Tiny subset containing only A, B, and a brand-new class HALO
    rng = np.random.default_rng(12)
    X_sub = np.concatenate(
        [
            np.clip(rng.normal(0.25, 0.04, size=(5, FEATURE_DIM)), 0.0, 1.0),
            np.clip(rng.normal(0.55, 0.04, size=(5, FEATURE_DIM)), 0.0, 1.0),
            np.clip(rng.normal(0.85, 0.04, size=(4, FEATURE_DIM)), 0.0, 1.0),
        ],
        axis=0,
    ).astype(np.float32)
    y_sub = np.array(["A"] * 5 + ["B"] * 5 + ["HALO"] * 4)

    sub_npz = tmp_path / "custom_small.npz"
    save_landmarks_npz(sub_npz, X_sub, y_sub)

    tflite_out = tmp_path / "artifacts" / "bisindo_alphabet_model_f32.tflite"
    labels_out = tmp_path / "artifacts" / "bisindo_alphabet_labels.json"
    tfjs_dir = tmp_path / "artifacts" / "tfjs"
    pub_dir = tmp_path / "public" / "models" / "bisindo"

    res = run_finetuning(
        npz_paths=[sub_npz],
        base_tflite_path=base_tflite,
        base_labels_path=base_labels,
        tflite_out=tflite_out,
        labels_out=labels_out,
        tfjs_dir=tfjs_dir,
        public_models_dir=pub_dir,
        target_classes=None,
        warmup_epochs=2,
        epochs=2,
        batch_size=4,
        jitter_copies=0,
        flip_swap=False,
        single_hand_ratio=0.0,
    )

    assert res.tflite_path.exists()
    assert (tfjs_dir / "model.json").exists()
    assert (tfjs_dir / "group1-shard1of1.bin").exists()
    assert (pub_dir / "model.json").exists()
    assert (pub_dir / "group1-shard1of1.bin").exists()
    assert (pub_dir / "labels.json").exists()
    assert (pub_dir / tflite_out.name).exists()

    new_labels = json.loads(res.labels_path.read_text(encoding="utf-8"))
    assert new_labels == [*labels_list, "HALO"]

    ft_keras = convert_tflite_mlp_to_keras(res.tflite_path)
    w_ft, b_ft = ft_keras.layers[-1].get_weights()
    assert np.allclose(w_ft[:, z_idx], w_base[:, z_idx], atol=1e-6)
    assert np.isclose(b_ft[z_idx], b_base[z_idx], atol=1e-6)


def test_augment_training_split_never_drops_slot1_from_single_hand_vectors():
    rng = np.random.default_rng(7)
    # Mix 10 single-hand vectors (slot 2 == 0) and 10 two-hand vectors
    X_1h = np.zeros((10, FEATURE_DIM), dtype=np.float32)
    X_1h[:, :42] = rng.uniform(0.1, 0.9, size=(10, 42)).astype(np.float32)
    X_2h = rng.uniform(0.1, 0.9, size=(10, FEATURE_DIM)).astype(np.float32)
    X = np.concatenate([X_1h, X_2h], axis=0)
    y = np.concatenate([np.zeros(10, dtype=np.int64), np.ones(10, dtype=np.int64)])

    X_aug, y_aug = augment_training_split(
        X, y, flip_swap=False, single_hand_ratio=0.3, jitter_copies=0, seed=42
    )
    # Every augmented row must have non-zero landmarks in slot 1 [0:42]
    assert np.all(np.any(np.abs(X_aug[:, :42]) > 1e-6, axis=1))
    # Only class 1 (two-hand) was eligible for single-hand fallback: 10+10 orig + 6 single = 26
    assert len(X_aug) == 26
    assert int(np.sum(y_aug == 0)) == 10


def test_run_v2_training_end_to_end_synthetic(tmp_path):
    rng = np.random.default_rng(15)
    classes = ["A", "C", "NOTHING"]
    hands_rows = []
    nh_rows = []
    y_rows = []
    groups_rows = []

    for i in range(12):
        # 2-hand class A
        h_a0 = rng.uniform(0.1, 0.35, size=(NUM_LANDMARKS, 2)).astype(np.float32)
        h_a1 = rng.uniform(0.6, 0.85, size=(NUM_LANDMARKS, 2)).astype(np.float32)
        hands_rows.append(np.stack([h_a0, h_a1], axis=0))
        nh_rows.append(2)
        y_rows.append("A")
        groups_rows.append(f"A_{i}")

        # 1-hand class C
        h_c = np.zeros((2, NUM_LANDMARKS, 2), dtype=np.float32)
        h_c[0] = rng.uniform(0.2, 0.7, size=(NUM_LANDMARKS, 2)).astype(np.float32)
        hands_rows.append(h_c)
        nh_rows.append(1)
        y_rows.append("C")
        groups_rows.append(f"C_{i}")

        # NOTHING (1 hand different cluster)
        h_n = np.zeros((2, NUM_LANDMARKS, 2), dtype=np.float32)
        h_n[0] = rng.uniform(0.75, 0.95, size=(NUM_LANDMARKS, 2)).astype(np.float32)
        hands_rows.append(h_n)
        nh_rows.append(1)
        y_rows.append("NOTHING")
        groups_rows.append(f"N_{i}")

    raw_npz = tmp_path / "raw_hands.npz"
    save_raw_hands_npz(
        raw_npz,
        np.stack(hands_rows, axis=0),
        np.array(nh_rows, dtype=np.int32),
        np.ones(len(y_rows), dtype=np.float32),
        np.array(y_rows),
        groups=np.array(groups_rows),
    )

    tflite_out = tmp_path / "artifacts" / "v2.tflite"
    labels_out = tmp_path / "artifacts" / "v2_labels.json"
    tfjs_dir = tmp_path / "artifacts" / "tfjs"
    pub_dir = tmp_path / "public" / "models" / "bisindo"

    res = run_v2_training(
        raw_hands_npz_paths=[raw_npz],
        classes=classes,
        tflite_out=tflite_out,
        labels_out=labels_out,
        tfjs_dir=tfjs_dir,
        public_models_dir=pub_dir,
        epochs=3,
        batch_size=16,
        aug_copies=1,
    )

    assert res.tflite_path.exists()
    assert (pub_dir / "model.json").exists()
    assert (pub_dir / "group1-shard1of1.bin").exists()
    assert (pub_dir / "v2.tflite").exists()
    payload = json.loads((pub_dir / "model.json").read_text(encoding="utf-8"))
    first_dense = [l for l in payload["modelTopology"]["model_config"]["config"]["layers"] if l["class_name"] == "Dense"][0]
    assert first_dense["config"]["batch_input_shape"] == [None, FEATURE_DIM_V2]
