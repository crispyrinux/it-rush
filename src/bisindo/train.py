"""Training and evaluation pipeline for the BISINDO Alphabet MLP model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from bisindo.dataset import (
    DatasetSplits,
    load_raw_hands_npz,
    merge_datasets_with_groups,
    split_dataset,
    summarize_dataset_npz,
)
from bisindo.features import (
    FEATURE_DIM_V2,
    TWO_HAND_LETTERS,
    _is_second_hand_present,
    add_landmark_jitter,
    build_bisindo_v2_features,
    flip_and_swap_features,
    make_single_hand_fallback,
    transform_hands_raw,
)
from bisindo.model import (
    build_alphabet_mlp,
    build_finetune_alphabet_model,
    export_tfjs_layers,
    export_tflite_f32,
    save_labels_json,
)


@dataclass(frozen=True)
class TrainResult:
    test_loss: float
    test_accuracy: float
    report: str
    tflite_path: Path
    labels_path: Path
    tfjs_model_path: Path | None = None


def augment_training_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    flip_swap: bool = True,
    single_hand_ratio: float = 0.15,
    jitter_std: float = 0.015,
    jitter_copies: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment training landmark features without touching validation/test splits."""
    rng = np.random.default_rng(seed)
    X_parts = [X_train]
    y_parts = [y_train]

    if flip_swap:
        X_flip = np.stack([flip_and_swap_features(x) for x in X_train], axis=0)
        X_parts.append(X_flip)
        y_parts.append(y_train)

    if single_hand_ratio > 0.0:
        two_hand_indices = np.array(
            [i for i, row in enumerate(X_train) if _is_second_hand_present(row)],
            dtype=np.int64,
        )
        if len(two_hand_indices) > 0:
            n_single = min(
                len(two_hand_indices),
                max(1, int(len(X_train) * single_hand_ratio)),
            )
            idx = rng.choice(two_hand_indices, size=n_single, replace=False)
            X_single = np.stack(
                [
                    make_single_hand_fallback(
                        X_train[i], keep="left" if rng.random() < 0.5 else "right"
                    )
                    for i in idx
                ],
                axis=0,
            )
            X_parts.append(X_single)
            y_parts.append(y_train[idx])

    for _ in range(jitter_copies):
        X_jit = np.stack(
            [add_landmark_jitter(x, std=jitter_std, rng=rng) for x in X_train], axis=0
        )
        X_parts.append(X_jit)
        y_parts.append(y_train)

    X_aug = np.concatenate(X_parts, axis=0)
    y_aug = np.concatenate(y_parts, axis=0)

    perm = rng.permutation(len(X_aug))
    return X_aug[perm], y_aug[perm]


def run_training(
    npz_paths: list[str | Path],
    classes: list[str],
    tflite_out: str | Path,
    labels_out: str | Path,
    tfjs_dir: str | Path | None = None,
    *,
    seed: int = 42,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 64,
    epochs: int = 120,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 15,
    reduce_lr_patience: int = 6,
    flip_swap: bool = True,
    single_hand_ratio: float = 0.15,
    jitter_std: float = 0.015,
    jitter_copies: int = 2,
) -> TrainResult:
    """Full training pipeline: merge -> group-isolated split -> augment -> fit -> eval -> export."""
    tf.keras.utils.set_random_seed(seed)

    X, y, groups = merge_datasets_with_groups(npz_paths, classes)
    splits: DatasetSplits = split_dataset(
        X, y, val_split=val_split, test_split=test_split, seed=seed, groups=groups
    )

    X_train_aug, y_train_aug = augment_training_split(
        splits.X_train,
        splits.y_train,
        flip_swap=flip_swap,
        single_hand_ratio=single_hand_ratio,
        jitter_std=jitter_std,
        jitter_copies=jitter_copies,
        seed=seed,
    )

    unique_classes = np.unique(y_train_aug)
    weights = compute_class_weight("balanced", classes=unique_classes, y=y_train_aug)
    class_weight = {int(c): float(w) for c, w in zip(unique_classes, weights)}

    model = build_alphabet_mlp(
        num_classes=len(classes),
        learning_rate=learning_rate,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-5,
        ),
    ]

    model.fit(
        X_train_aug,
        y_train_aug,
        validation_data=(splits.X_val, splits.y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(splits.X_test, splits.y_test, verbose=0)
    y_pred = np.argmax(model.predict(splits.X_test, verbose=0), axis=1)

    present_labels = sorted(np.unique(splits.y_test).tolist())
    present_names = [classes[i] for i in present_labels]
    report = classification_report(
        splits.y_test,
        y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    )

    tflite_path = export_tflite_f32(model, tflite_out)
    labels_path = save_labels_json(classes, labels_out)
    tfjs_model_path = export_tfjs_layers(model, classes, tfjs_dir) if tfjs_dir else None

    return TrainResult(
        test_loss=float(test_loss),
        test_accuracy=float(test_acc),
        report=report,
        tflite_path=tflite_path,
        labels_path=labels_path,
        tfjs_model_path=tfjs_model_path,
    )


def _safe_split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    val_split: float,
    test_split: float,
    seed: int,
) -> DatasetSplits:
    """Split dataset with group-isolation, falling back cleanly for very small sets."""
    try:
        return split_dataset(
            X, y, val_split=val_split, test_split=test_split, seed=seed, groups=groups
        )
    except ValueError:
        n = len(X)
        if n < 3:
            return DatasetSplits(
                X_train=X, y_train=y, X_val=X, y_val=y, X_test=X, y_test=y
            )
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_test = max(1, min(n - 2, int(round(n * test_split))))
        n_val = max(1, min(n - n_test - 1, int(round(n * val_split))))
        n_train = n - n_test - n_val
        return DatasetSplits(
            X_train=X[perm[:n_train]],
            y_train=y[perm[:n_train]],
            X_val=X[perm[n_train : n_train + n_val]],
            y_val=y[perm[n_train : n_train + n_val]],
            X_test=X[perm[n_train + n_val :]],
            y_test=y[perm[n_train + n_val :]],
        )


def _compile_classifier(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def run_finetuning(
    npz_paths: list[str | Path],
    base_tflite_path: str | Path,
    base_labels_path: str | Path,
    tflite_out: str | Path,
    labels_out: str | Path,
    tfjs_dir: str | Path | None = None,
    public_models_dir: str | Path | None = None,
    target_classes: list[str] | None = None,
    *,
    seed: int = 42,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 32,
    warmup_epochs: int = 5,
    warmup_lr: float = 5e-4,
    epochs: int = 40,
    learning_rate: float = 1e-4,
    l2_sp_alpha: float = 1e-2,
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 4,
    flip_swap: bool = True,
    single_hand_ratio: float = 0.15,
    jitter_std: float = 0.015,
    jitter_copies: int = 2,
) -> TrainResult:
    """Two-stage anti-forgetting fine-tuning on top of pretrained KawanIsyarat TFLite."""
    tf.keras.utils.set_random_seed(seed)

    base_classes: list[str] = json.loads(
        Path(base_labels_path).read_text(encoding="utf-8")
    )
    if target_classes is None:
        discovered = summarize_dataset_npz(npz_paths)
        resolved_classes = list(base_classes) + [
            lbl for lbl in sorted(discovered) if lbl not in base_classes
        ]
    else:
        resolved_classes = list(target_classes)

    X, y, groups = merge_datasets_with_groups(npz_paths, resolved_classes)
    splits = _safe_split_dataset(X, y, groups, val_split, test_split, seed)

    X_train_aug, y_train_aug = augment_training_split(
        splits.X_train,
        splits.y_train,
        flip_swap=flip_swap,
        single_hand_ratio=single_hand_ratio,
        jitter_std=jitter_std,
        jitter_copies=jitter_copies,
        seed=seed,
    )

    unique_classes = np.unique(y_train_aug)
    weights = compute_class_weight("balanced", classes=unique_classes, y=y_train_aug)
    class_weight = {int(c): float(w) for c, w in zip(unique_classes, weights)}

    model, shared_map, (w_init, b_init) = build_finetune_alphabet_model(
        base_tflite_path=base_tflite_path,
        base_classes=base_classes,
        target_classes=resolved_classes,
        learning_rate=learning_rate,
        l2_sp_alpha=l2_sp_alpha,
        seed=seed,
    )

    if warmup_epochs > 0:
        for layer in model.layers[:-1]:
            layer.trainable = False
        model.layers[-1].trainable = True
        _compile_classifier(model, warmup_lr)
        model.fit(
            X_train_aug,
            y_train_aug,
            validation_data=(splits.X_val, splits.y_val),
            epochs=warmup_epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=1,
        )

    if epochs > 0:
        for layer in model.layers:
            layer.trainable = True
        _compile_classifier(model, learning_rate)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-5,
            ),
        ]
        model.fit(
            X_train_aug,
            y_train_aug,
            validation_data=(splits.X_val, splits.y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

    # Restore pretrained classifier columns for classes absent from fine-tuning data
    present_targets = set(unique_classes.tolist())
    out_layer = model.layers[-1]
    w_final, b_final = out_layer.get_weights()
    restored_any = False
    for t_idx in shared_map:
        if t_idx not in present_targets:
            w_final[:, t_idx] = w_init[:, t_idx]
            b_final[t_idx] = b_init[t_idx]
            restored_any = True
    if restored_any:
        out_layer.set_weights([w_final, b_final])

    test_loss, test_acc = model.evaluate(splits.X_test, splits.y_test, verbose=0)
    y_pred = np.argmax(model.predict(splits.X_test, verbose=0), axis=1)

    present_test = sorted(np.unique(splits.y_test).tolist())
    present_names = [resolved_classes[i] for i in present_test]
    report = classification_report(
        splits.y_test,
        y_pred,
        labels=present_test,
        target_names=present_names,
        zero_division=0,
    )

    tflite_path = export_tflite_f32(model, tflite_out)
    labels_path = save_labels_json(resolved_classes, labels_out)
    tfjs_model_path = export_tfjs_layers(model, resolved_classes, tfjs_dir) if tfjs_dir else None

    if public_models_dir is not None:
        pub_dir = Path(public_models_dir)
        pub_dir.mkdir(parents=True, exist_ok=True)
        export_tfjs_layers(model, resolved_classes, pub_dir)
        pub_tflite = pub_dir / tflite_path.name
        if pub_tflite.resolve() != tflite_path.resolve():
            shutil.copy2(tflite_path, pub_tflite)

    return TrainResult(
        test_loss=float(test_loss),
        test_accuracy=float(test_acc),
        report=report,
        tflite_path=tflite_path,
        labels_path=labels_path,
        tfjs_model_path=tfjs_model_path,
    )


def _merge_raw_hands_datasets(
    npz_paths: list[str | Path],
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    all_h: list[np.ndarray] = []
    all_nh: list[np.ndarray] = []
    all_ar: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_g: list[np.ndarray] = []

    for src_idx, p in enumerate(npz_paths):
        path = Path(p)
        if not path.exists():
            continue
        hands, num_hands, aspect_ratios, y_str, groups = load_raw_hands_npz(path)
        valid_mask = np.array(
            [lbl in label_to_idx and int(nh) > 0 for lbl, nh in zip(y_str, num_hands)],
            dtype=bool,
        )
        if not np.any(valid_mask):
            continue

        if groups is None:
            file_groups = np.array([f"raw{src_idx}_{i}" for i in range(len(hands))], dtype=str)
        else:
            file_groups = np.array([f"raw{src_idx}_{g}" for g in groups], dtype=str)

        all_h.append(hands[valid_mask])
        all_nh.append(num_hands[valid_mask])
        all_ar.append(aspect_ratios[valid_mask])
        all_y.append(np.array([label_to_idx[lbl] for lbl in y_str[valid_mask]], dtype=np.int64))
        all_g.append(file_groups[valid_mask])

    if not all_h:
        raise ValueError("No valid raw_hands samples found across provided NPZ paths.")

    return (
        np.concatenate(all_h, axis=0),
        np.concatenate(all_nh, axis=0),
        np.concatenate(all_ar, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_g, axis=0),
    )


def _extract_v2_feature_matrix(
    hands: np.ndarray,
    num_hands: np.ndarray,
    aspect_ratios: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for idx in indices:
        nh = int(num_hands[idx])
        h_list = [hands[idx, 0]] if nh <= 1 else [hands[idx, 0], hands[idx, 1]]
        rows.append(build_bisindo_v2_features(h_list, aspect_ratio=float(aspect_ratios[idx])))
    return np.stack(rows, axis=0).astype(np.float32)


def augment_raw_hands_split(
    hands: np.ndarray,
    num_hands: np.ndarray,
    aspect_ratios: np.ndarray,
    y: np.ndarray,
    *,
    two_hand_class_indices: set[int] | None = None,
    aug_copies: int = 3,
    single_hand_ratio: float = 0.20,
    close_dx_swap_thresh: float = 0.30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate aspect/scale/rotation-robust 222-dim training features from raw landmarks."""
    rng = np.random.default_rng(seed)
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    aspect_choices = (1.0, 4.0 / 3.0, 16.0 / 9.0, 0.85)
    indices_2h_by_cls: dict[int, list[int]] = {}

    for i in range(len(hands)):
        nh = int(num_hands[i])
        if nh <= 0:
            continue
        cls_idx = int(y[i])
        ar = float(aspect_ratios[i])
        h_list = [hands[i, 0]] if nh == 1 else [hands[i, 0], hands[i, 1]]

        if nh >= 2:
            indices_2h_by_cls.setdefault(cls_idx, []).append(i)

        for fl in (False, True):
            h_clean, ar_clean = transform_hands_raw(h_list, ar, flip=fl)
            v2_clean = build_bisindo_v2_features(h_clean, aspect_ratio=ar_clean)
            X_rows.append(v2_clean)
            y_rows.append(cls_idx)

            if nh >= 2 and abs(float(v2_clean[218])) < close_dx_swap_thresh:
                X_rows.append(
                    build_bisindo_v2_features(h_clean, aspect_ratio=ar_clean, swap_slots=True)
                )
                y_rows.append(cls_idx)

            for _ in range(aug_copies):
                rot = float(rng.uniform(-16.0, 16.0))
                sc = float(rng.uniform(0.30, 1.25))
                am = float(rng.choice(aspect_choices))
                h_aug, ar_aug = transform_hands_raw(
                    h_list,
                    ar,
                    flip=fl,
                    rot_deg=rot,
                    scale_dist=sc,
                    aspect_mult=am,
                    noise_std=0.015,
                    rng=rng,
                )
                v2_aug = build_bisindo_v2_features(h_aug, aspect_ratio=ar_aug)
                X_rows.append(v2_aug)
                y_rows.append(cls_idx)
                if nh >= 2 and abs(float(v2_aug[218])) < close_dx_swap_thresh:
                    X_rows.append(
                        build_bisindo_v2_features(h_aug, aspect_ratio=ar_aug, swap_slots=True)
                    )
                    y_rows.append(cls_idx)

        if nh >= 2 and single_hand_ratio > 0.0 and rng.random() < single_hand_ratio:
            kept = [h_list[0]] if rng.random() < 0.5 else [h_list[1]]
            h_occ, ar_occ = transform_hands_raw(
                kept,
                ar,
                flip=bool(rng.random() < 0.5),
                rot_deg=float(rng.uniform(-12.0, 12.0)),
                scale_dist=float(rng.uniform(0.45, 1.15)),
                aspect_mult=float(rng.choice(aspect_choices)),
                noise_std=0.012,
                rng=rng,
            )
            X_rows.append(build_bisindo_v2_features(h_occ, aspect_ratio=ar_occ))
            y_rows.append(cls_idx)

    target_2h = two_hand_class_indices or set(indices_2h_by_cls.keys())
    counts_2h = [len(indices_2h_by_cls.get(c, [])) for c in target_2h if indices_2h_by_cls.get(c)]
    if counts_2h:
        median_2h = int(np.median(counts_2h))
        for c in sorted(target_2h):
            cls_indices = indices_2h_by_cls.get(c, [])
            if not cls_indices or len(cls_indices) >= median_2h:
                continue
            deficit = median_2h - len(cls_indices)
            extra_copies = max(2, deficit * 2)
            for _ in range(extra_copies):
                src_i = int(rng.choice(cls_indices))
                h_list = [hands[src_i, 0], hands[src_i, 1]]
                h_aug, ar_aug = transform_hands_raw(
                    h_list,
                    float(aspect_ratios[src_i]),
                    flip=bool(rng.random() < 0.5),
                    rot_deg=float(rng.uniform(-16.0, 16.0)),
                    scale_dist=float(rng.uniform(0.30, 1.20)),
                    aspect_mult=float(rng.choice(aspect_choices)),
                    noise_std=0.015,
                    rng=rng,
                )
                v2_bal = build_bisindo_v2_features(h_aug, aspect_ratio=ar_aug)
                X_rows.append(v2_bal)
                y_rows.append(c)

    X_out = np.stack(X_rows, axis=0).astype(np.float32)
    y_out = np.array(y_rows, dtype=np.int64)
    perm = rng.permutation(len(X_out))
    return X_out[perm], y_out[perm]


def run_v2_training(
    raw_hands_npz_paths: list[str | Path],
    classes: list[str],
    tflite_out: str | Path,
    labels_out: str | Path,
    tfjs_dir: str | Path | None = None,
    public_models_dir: str | Path | None = None,
    *,
    seed: int = 42,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 64,
    epochs: int = 120,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 15,
    reduce_lr_patience: int = 6,
    aug_copies: int = 3,
    single_hand_ratio: float = 0.20,
) -> TrainResult:
    """Full V2 222-dim pipeline: merge raw hands -> split -> camera augment -> fit -> eval -> export."""
    tf.keras.utils.set_random_seed(seed)

    hands, num_hands, aspect_ratios, y_idx, groups = _merge_raw_hands_datasets(
        raw_hands_npz_paths, classes
    )
    idx_col = np.arange(len(y_idx), dtype=np.float32).reshape(-1, 1)
    splits = _safe_split_dataset(idx_col, y_idx, groups, val_split, test_split, seed)

    tr_idx = splits.X_train[:, 0].astype(int)
    va_idx = splits.X_val[:, 0].astype(int)
    te_idx = splits.X_test[:, 0].astype(int)

    cls_to_idx = {lbl: idx for idx, lbl in enumerate(classes)}
    two_hand_class_indices = {
        cls_to_idx[lbl] for lbl in TWO_HAND_LETTERS if lbl in cls_to_idx
    }

    X_train_aug, y_train_aug = augment_raw_hands_split(
        hands[tr_idx],
        num_hands[tr_idx],
        aspect_ratios[tr_idx],
        y_idx[tr_idx],
        two_hand_class_indices=two_hand_class_indices,
        aug_copies=aug_copies,
        single_hand_ratio=single_hand_ratio,
        seed=seed,
    )
    X_val = _extract_v2_feature_matrix(hands, num_hands, aspect_ratios, va_idx)
    y_val = y_idx[va_idx]
    X_test = _extract_v2_feature_matrix(hands, num_hands, aspect_ratios, te_idx)
    y_test = y_idx[te_idx]

    unique_classes = np.unique(y_train_aug)
    weights = compute_class_weight("balanced", classes=unique_classes, y=y_train_aug)
    class_weight = {int(c): float(w) for c, w in zip(unique_classes, weights)}

    model = build_alphabet_mlp(
        num_classes=len(classes),
        feature_dim=FEATURE_DIM_V2,
        learning_rate=learning_rate,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-5,
        ),
    ]
    model.fit(
        X_train_aug,
        y_train_aug,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    present_labels = sorted(np.unique(y_test).tolist())
    present_names = [classes[i] for i in present_labels]
    report = classification_report(
        y_test,
        y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    )

    # Group metrics (1-hand vs 2-hand + simulated webcam stress)
    te_nh = num_hands[te_idx]
    is_2h_cls = np.array([classes[i] in TWO_HAND_LETTERS for i in y_test], dtype=bool)
    is_1h_cls = ~is_2h_cls
    summary_lines = ["\n=== V2 Group Summary ==="]
    if np.any(is_1h_cls):
        acc_1h = float((y_pred[is_1h_cls] == y_test[is_1h_cls]).mean())
        summary_lines.append(f"  1-HAND Clean ({int(np.sum(is_1h_cls))}): {acc_1h:.2%}")
    if np.any(is_2h_cls):
        acc_2h = float((y_pred[is_2h_cls] == y_test[is_2h_cls]).mean())
        m_2d = is_2h_cls & (te_nh >= 2)
        m_1d = is_2h_cls & (te_nh < 2)
        acc_2h_2d = float((y_pred[m_2d] == y_test[m_2d]).mean()) if np.any(m_2d) else 0.0
        acc_2h_1d = float((y_pred[m_1d] == y_test[m_1d]).mean()) if np.any(m_1d) else 0.0
        summary_lines.append(
            f"  2-HAND Clean ({int(np.sum(is_2h_cls))}): {acc_2h:.2%} "
            f"(2h-det={acc_2h_2d:.2%} n={int(np.sum(m_2d))}, "
            f"1h-det={acc_2h_1d:.2%} n={int(np.sum(m_1d))})"
        )

    rng_stress = np.random.default_rng(seed + 1000)
    stress_rows: list[np.ndarray] = []
    for idx in te_idx:
        nh = int(num_hands[idx])
        h_list = [hands[idx, 0]] if nh <= 1 else [hands[idx, 0], hands[idx, 1]]
        h_s, ar_s = transform_hands_raw(
            h_list,
            float(aspect_ratios[idx]),
            flip=bool(rng_stress.random() < 0.5),
            rot_deg=float(rng_stress.uniform(-14.0, 14.0)),
            scale_dist=float(rng_stress.uniform(0.30, 0.55)),
            aspect_mult=4.0 / 3.0,
            noise_std=0.012,
            rng=rng_stress,
        )
        stress_rows.append(build_bisindo_v2_features(h_s, aspect_ratio=ar_s))
    X_stress = np.stack(stress_rows, axis=0).astype(np.float32)
    y_stress_pred = np.argmax(model.predict(X_stress, verbose=0), axis=1)
    acc_stress_all = float((y_stress_pred == y_test).mean())
    acc_stress_2h = (
        float((y_stress_pred[is_2h_cls] == y_test[is_2h_cls]).mean())
        if np.any(is_2h_cls)
        else acc_stress_all
    )
    summary_lines.append(
        f"  Webcam Stress (4:3, 0.30-0.55x, +-14deg, flip): ALL={acc_stress_all:.2%}, 2-HAND={acc_stress_2h:.2%}"
    )
    report = report + "\n".join(summary_lines)

    tflite_path = export_tflite_f32(model, tflite_out)
    labels_path = save_labels_json(classes, labels_out)
    tfjs_model_path = export_tfjs_layers(model, classes, tfjs_dir) if tfjs_dir else None

    if public_models_dir is not None:
        pub_dir = Path(public_models_dir)
        pub_dir.mkdir(parents=True, exist_ok=True)
        export_tfjs_layers(model, classes, pub_dir)
        pub_tflite = pub_dir / tflite_path.name
        if pub_tflite.resolve() != tflite_path.resolve():
            shutil.copy2(tflite_path, pub_tflite)

    return TrainResult(
        test_loss=float(test_loss),
        test_accuracy=float(test_acc),
        report=report,
        tflite_path=tflite_path,
        labels_path=labels_path,
        tfjs_model_path=tfjs_model_path,
    )
