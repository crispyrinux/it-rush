"""Compare WL-BISINDO training methods on the 95-clip dataset without touching production models.

Evaluates 3 methods using leak-free 5-fold StratifiedGroupKFold (grouped by original clip):
  1. Frozen Conformer + Linear Head (current production method)
  2. End-to-End Conformer (trainable Conv1D + MHSA + FFN, label smoothing + MixUp)
  3. End-to-End Geometric Conformer (adds hand/body distances, angles, openness + Conformer)
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from bisindo.word_level import (
    WL_RAW_FEATURE_DIM,
    WL_SEQUENCE_LEN,
    extract_wl_backbone_embeddings,
    load_wl_sequences_npz,
    train_wl_custom_head,
)


def extract_geometric_features(X_seq: np.ndarray) -> np.ndarray:
    """Derive scale-robust distances, openness, and limb angles from `(N, 30, 300)` sequences."""
    pos = X_seq[:, :, :WL_RAW_FEATURE_DIM]  # (N, 30, 100)
    rh = pos[:, :, 0:42].reshape(pos.shape[0], pos.shape[1], 21, 2)
    lh = pos[:, :, 42:84].reshape(pos.shape[0], pos.shape[1], 21, 2)
    pose = pos[:, :, 84:98].reshape(pos.shape[0], pos.shape[1], 7, 2)
    has_r = pos[:, :, 98:99]
    has_l = pos[:, :, 99:100]

    nose = pose[:, :, 0, :]       # (N, 30, 2)
    l_sh = pose[:, :, 1, :]
    r_sh = pose[:, :, 2, :]
    l_el = pose[:, :, 5, :]
    r_el = pose[:, :, 6, :]

    def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.norm(a - b, axis=-1, keepdims=True)

    def unit_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        v = b - a
        n = np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-4)
        return v / n

    # Key hand landmarks: 0=wrist, 4=thumb_tip, 8=index_tip, 9=middle_mcp, 12=middle_tip, 20=pinky_tip
    r_wrist, l_wrist = rh[:, :, 0, :], lh[:, :, 0, :]
    r_palm, l_palm = rh[:, :, 9, :], lh[:, :, 9, :]
    r_thumb, l_thumb = rh[:, :, 4, :], lh[:, :, 4, :]
    r_idx, l_idx = rh[:, :, 8, :], lh[:, :, 8, :]
    r_mid, l_mid = rh[:, :, 12, :], lh[:, :, 12, :]
    r_pinky, l_pinky = rh[:, :, 20, :], lh[:, :, 20, :]

    feats = [
        # Distances from hands to face & shoulders
        dist(r_wrist, nose) * has_r,
        dist(l_wrist, nose) * has_l,
        dist(r_idx, nose) * has_r,
        dist(l_idx, nose) * has_l,
        dist(r_wrist, r_sh) * has_r,
        dist(l_wrist, l_sh) * has_l,
        dist(r_wrist, l_sh) * has_r,
        dist(l_wrist, r_sh) * has_l,
        # Inter-hand distances
        dist(r_wrist, l_wrist) * has_r * has_l,
        dist(r_idx, l_idx) * has_r * has_l,
        dist(r_palm, l_palm) * has_r * has_l,
        # Hand openness / finger spread
        dist(r_thumb, r_idx) * has_r,
        dist(r_thumb, r_pinky) * has_r,
        dist(r_wrist, r_mid) * has_r,
        dist(l_thumb, l_idx) * has_l,
        dist(l_thumb, l_pinky) * has_l,
        dist(l_wrist, l_mid) * has_l,
        # Forearm & palm orientation unit vectors (cos, sin)
        unit_vec(r_el, r_wrist) * has_r,
        unit_vec(l_el, l_wrist) * has_l,
        unit_vec(r_wrist, r_palm) * has_r,
        unit_vec(l_wrist, l_palm) * has_l,
    ]
    geo = np.concatenate(feats, axis=-1).astype(np.float32)  # (N, 30, 25)
    geo_vel = np.zeros_like(geo)
    geo_vel[:, 1:, :] = geo[:, 1:, :] - geo[:, :-1, :]
    return np.concatenate([X_seq, geo, geo_vel], axis=-1).astype(np.float32)


def _conformer_block(x: tf.Tensor, dim: int, num_heads: int, dropout: float) -> tf.Tensor:
    # Causal-like depthwise conv module
    h = tf.keras.layers.LayerNormalization()(x)
    h = tf.keras.layers.Dense(dim * 2, activation="swish")(h)
    h = tf.keras.layers.DepthwiseConv1D(kernel_size=7, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Activation("swish")(h)
    h = tf.keras.layers.Dense(dim)(h)
    h = tf.keras.layers.Dropout(dropout)(h)
    x = tf.keras.layers.Add()([x, h])

    # Multi-head self-attention module
    h_att = tf.keras.layers.LayerNormalization()(x)
    h_att = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=dim // num_heads, dropout=dropout
    )(h_att, h_att)
    h_att = tf.keras.layers.Dropout(dropout)(h_att)
    x = tf.keras.layers.Add()([x, h_att])

    # Feed-forward module
    h_ff = tf.keras.layers.LayerNormalization()(x)
    h_ff = tf.keras.layers.Dense(dim * 2, activation="swish")(h_ff)
    h_ff = tf.keras.layers.Dropout(dropout)(h_ff)
    h_ff = tf.keras.layers.Dense(dim)(h_ff)
    h_ff = tf.keras.layers.Dropout(dropout)(h_ff)
    return tf.keras.layers.Add()([x, h_ff])


def build_e2e_conformer(
    seq_len: int,
    feat_dim: int,
    num_classes: int,
    *,
    dim: int = 64,
    num_heads: int = 2,
    num_blocks: int = 2,
    dropout: float = 0.25,
) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(seq_len, feat_dim), dtype=tf.float32)
    x = tf.keras.layers.Dense(dim, use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    for _ in range(num_blocks):
        x = _conformer_block(x, dim=dim, num_heads=num_heads, dropout=dropout)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inp, outputs=out)


def train_e2e_with_mixup(
    X_train: np.ndarray,
    y_train_idx: np.ndarray,
    num_classes: int,
    *,
    epochs: int = 70,
    batch_size: int = 32,
    seed: int = 42,
) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    model = build_e2e_conformer(X_train.shape[1], X_train.shape[2], num_classes)
    steps_per_epoch = max(1, int(np.ceil(len(X_train) / batch_size)))
    lr_sched = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=2e-3,
        decay_steps=epochs * steps_per_epoch,
        alpha=0.05,
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    y_oh = tf.keras.utils.to_categorical(y_train_idx, num_classes=num_classes).astype(np.float32)
    cls_weights_arr = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=y_train_idx
    )
    sample_weights_base = cls_weights_arr[y_train_idx].astype(np.float32)

    n = len(X_train)
    for _ in range(epochs):
        perm = rng.permutation(n)
        X_shuf = X_train[perm]
        y_shuf = y_oh[perm]
        w_shuf = sample_weights_base[perm]

        # MixUp on 50% of batches
        if rng.random() < 0.5:
            perm2 = rng.permutation(n)
            lam = rng.beta(0.3, 0.3, size=(n, 1, 1)).astype(np.float32)
            X_ep = lam * X_shuf + (1.0 - lam) * X_shuf[perm2]
            lam_y = lam[:, 0, :]
            y_ep = lam_y * y_shuf + (1.0 - lam_y) * y_shuf[perm2]
            w_ep = lam[:, 0, 0] * w_shuf + (1.0 - lam[:, 0, 0]) * w_shuf[perm2]
        else:
            X_ep, y_ep, w_ep = X_shuf, y_shuf, w_shuf

        model.fit(
            X_ep,
            y_ep,
            sample_weight=w_ep,
            batch_size=batch_size,
            epochs=1,
            verbose=0,
        )

    return model


def summarize_results(
    name: str,
    y_true: np.ndarray,
    probs_all: np.ndarray,
    classes: list[str],
    train_s: float,
    infer_ms: float,
    params: int,
) -> dict[str, Any]:
    preds = np.argmax(probs_all, axis=1)
    acc = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, average="macro"))
    confs = probs_all[np.arange(len(y_true)), preds]
    correct_mask = preds == y_true
    mean_conf_ok = float(np.mean(confs[correct_mask])) if np.any(correct_mask) else 0.0

    per_class: dict[str, str] = {}
    mistakes: list[str] = []
    for c_idx, cls_name in enumerate(classes):
        mask = y_true == c_idx
        c_ok = int(np.sum(preds[mask] == c_idx))
        c_tot = int(np.sum(mask))
        per_class[cls_name] = f"{c_ok}/{c_tot} ({c_ok / max(1, c_tot):.0%})"
        for i in np.where(mask & (preds != c_idx))[0]:
            mistakes.append(f"{cls_name}->{classes[int(preds[i])]} ({confs[i]:.0%})")

    return {
        "method": name,
        "oof_accuracy": acc,
        "correct": int(np.sum(correct_mask)),
        "total": int(len(y_true)),
        "macro_f1": f1,
        "mean_correct_conf": mean_conf_ok,
        "params": params,
        "train_seconds_5fold": round(train_s, 1),
        "infer_ms_per_clip": round(infer_ms, 2),
        "per_class": per_class,
        "mistakes": mistakes,
    }


def main() -> int:
    npz_path = Path("data/processed/wl_custom_landmarks.npz")
    tflite_path = Path("artifacts/bisindo_wl_model.tflite")
    X_all, y_str, g_all = load_wl_sequences_npz(npz_path)

    is_clean = np.array(["_aug" not in str(grp) for grp in g_all])
    base_groups = np.array([str(grp).split("_aug")[0] for grp in g_all])
    classes = sorted(set(y_str.tolist()))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([label_to_idx[str(s)] for s in y_str], dtype=np.int64)

    clean_indices = np.where(is_clean)[0]
    y_clean_true = y_idx[clean_indices]

    # Pre-extract frozen embeddings once for Method 1 (backbone is fixed)
    emb_all, _, _ = extract_wl_backbone_embeddings(tflite_path, X_all)
    # Pre-compute geometric features once for Method 3
    X_geo_all = extract_geometric_features(X_all)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(X_all, y_idx, base_groups))

    # ── Method 1: Frozen Conformer + Linear Head ──────────────────────────
    t0 = time.perf_counter()
    oof_probs_m1 = np.zeros((len(clean_indices), len(classes)), dtype=np.float32)
    clean_pos_map = {int(orig_i): pos for pos, orig_i in enumerate(clean_indices)}
    infer_times_m1: list[float] = []

    for tr_idx, va_idx in splits:
        w, b, _ = train_wl_custom_head(
            tflite_path,
            classes,
            X_all[tr_idx],
            y_str[tr_idx],
            epochs=80,
            learning_rate=3e-3,
            l2_reg=1e-4,
            seed=42,
        )
        va_clean = [int(i) for i in va_idx if is_clean[i]]
        ti0 = time.perf_counter()
        logits = emb_all[va_clean] @ w.T + b
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        infer_times_m1.append((time.perf_counter() - ti0) * 1000.0 / max(1, len(va_clean)))
        for row_i, orig_i in enumerate(va_clean):
            oof_probs_m1[clean_pos_map[orig_i]] = probs[row_i]
    res_m1 = summarize_results(
        "1. Frozen Conformer + Linear Head (Saat Ini)",
        y_clean_true,
        oof_probs_m1,
        classes,
        train_s=time.perf_counter() - t0,
        infer_ms=float(np.mean(infer_times_m1)),
        params=11 * 128 + 11,
    )

    # ── Method 2: End-to-End Conformer (300-dim) ──────────────────────────
    t0 = time.perf_counter()
    oof_probs_m2 = np.zeros((len(clean_indices), len(classes)), dtype=np.float32)
    infer_times_m2: list[float] = []
    m2_params = 0

    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        model = train_e2e_with_mixup(X_all[tr_idx], y_idx[tr_idx], len(classes), epochs=65, seed=42 + fold_i)
        m2_params = int(model.count_params())
        va_clean = [int(i) for i in va_idx if is_clean[i]]
        ti0 = time.perf_counter()
        probs = model.predict(X_all[va_clean], verbose=0)
        infer_times_m2.append((time.perf_counter() - ti0) * 1000.0 / max(1, len(va_clean)))
        for row_i, orig_i in enumerate(va_clean):
            oof_probs_m2[clean_pos_map[orig_i]] = probs[row_i]
    res_m2 = summarize_results(
        "2. End-to-End Conformer + MixUp + LabelSmooth (300-dim)",
        y_clean_true,
        oof_probs_m2,
        classes,
        train_s=time.perf_counter() - t0,
        infer_ms=float(np.mean(infer_times_m2)),
        params=m2_params,
    )

    # ── Method 3: End-to-End Geometric Conformer (350-dim) ────────────────
    t0 = time.perf_counter()
    oof_probs_m3 = np.zeros((len(clean_indices), len(classes)), dtype=np.float32)
    infer_times_m3: list[float] = []
    m3_params = 0

    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        model = train_e2e_with_mixup(X_geo_all[tr_idx], y_idx[tr_idx], len(classes), epochs=65, seed=142 + fold_i)
        m3_params = int(model.count_params())
        va_clean = [int(i) for i in va_idx if is_clean[i]]
        ti0 = time.perf_counter()
        probs = model.predict(X_geo_all[va_clean], verbose=0)
        infer_times_m3.append((time.perf_counter() - ti0) * 1000.0 / max(1, len(va_clean)))
        for row_i, orig_i in enumerate(va_clean):
            oof_probs_m3[clean_pos_map[orig_i]] = probs[row_i]
    res_m3 = summarize_results(
        "3. End-to-End Geometric Conformer (Jarak+Sudut+MixUp, 350-dim)",
        y_clean_true,
        oof_probs_m3,
        classes,
        train_s=time.perf_counter() - t0,
        infer_ms=float(np.mean(infer_times_m3)),
        params=m3_params,
    )

    report = {"dataset": "95 clean clips (5-fold group out-of-fold)", "results": [res_m1, res_m2, res_m3]}
    out_path = Path("artifacts/experiments/wl_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n================ 5-FOLD OUT-OF-SAMPLE COMPARISON ================")
    for r in report["results"]:
        print(
            f"\n{r['method']}\n"
            f"  OOF Accuracy : {r['correct']}/{r['total']} ({r['oof_accuracy']:.2%})\n"
            f"  Macro F1     : {r['macro_f1']:.4f}\n"
            f"  Mean Conf    : {r['mean_correct_conf']:.1%}\n"
            f"  Params       : {r['params']:,}\n"
            f"  5-Fold Time  : {r['train_seconds_5fold']}s\n"
            f"  Mistakes ({len(r['mistakes'])}): {r['mistakes']}"
        )
    print(f"\nSaved full comparison -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
