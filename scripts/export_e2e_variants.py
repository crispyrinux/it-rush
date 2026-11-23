"""Train and export End-to-End Conformer variants (B & C) for live A/B/C testing in /bisindo.

Does NOT overwrite the existing Frozen Conformer + Linear Head model in `public/models/bisindo/`.
Writes variants into `public/models/bisindo/variants/<id>/` and registers them in
`public/models/bisindo/wl_variants.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import tensorflow as tf

from bisindo.custom_words import write_wl_labels_json
from bisindo.word_level import load_wl_sequences_npz
from scripts.compare_wl_methods import extract_geometric_features, train_e2e_with_mixup


def _fold_bn(bn_layer: tf.keras.layers.BatchNormalization) -> tuple[np.ndarray, np.ndarray]:
    gamma, beta, mean, var = [np.asarray(w, dtype=np.float32) for w in bn_layer.get_weights()]
    scale = gamma / np.sqrt(var + float(bn_layer.epsilon))
    shift = beta - mean * scale
    return scale.astype("<f4"), shift.astype("<f4")


def export_e2e_conformer_to_browser(
    model: tf.keras.Model,
    classes: list[str],
    out_dir: Path,
    *,
    feature_mode: str,
) -> Path:
    """Serialize `build_e2e_conformer` weights into `wl_e2e_model.json` + `wl_e2e_weights.bin`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    layers_by_type: dict[str, list[tf.keras.layers.Layer]] = {}
    for layer in model.layers:
        cls = type(layer).__name__
        layers_by_type.setdefault(cls, []).append(layer)

    denses = layers_by_type["Dense"]
    bns = layers_by_type["BatchNormalization"]
    lns = layers_by_type["LayerNormalization"]
    dwconvs = layers_by_type["DepthwiseConv1D"]
    mhas = layers_by_type["MultiHeadAttention"]

    num_blocks = len(mhas)
    dim = int(denses[0].units)
    chunks: list[bytes] = []
    tensors: dict[str, dict[str, Any]] = {}
    offset = 0

    def add_tensor(name: str, arr: np.ndarray) -> str:
        nonlocal offset
        c_arr = np.ascontiguousarray(arr, dtype="<f4")
        raw = c_arr.tobytes(order="C")
        tensors[name] = {
            "shape": [int(s) for s in c_arr.shape],
            "dtype": "float32",
            "byteOffset": offset,
            "byteLength": len(raw),
        }
        chunks.append(raw)
        offset += len(raw)
        return name

    def add_dense(prefix: str, dense_layer: tf.keras.layers.Dense) -> dict[str, str]:
        weights = dense_layer.get_weights()
        w = np.asarray(weights[0], dtype=np.float32).T
        spec = {"w": add_tensor(f"{prefix}.w", w)}
        if len(weights) > 1:
            b = np.asarray(weights[1], dtype=np.float32)
            spec["b"] = add_tensor(f"{prefix}.b", b)
        return spec

    def add_ln(prefix: str, ln_layer: tf.keras.layers.LayerNormalization) -> dict[str, str]:
        g, b = [np.asarray(w, dtype=np.float32) for w in ln_layer.get_weights()]
        return {
            "gamma": add_tensor(f"{prefix}.gamma", g),
            "beta": add_tensor(f"{prefix}.beta", b),
        }

    def add_bn(prefix: str, bn_layer: tf.keras.layers.BatchNormalization) -> dict[str, str]:
        scale, shift = _fold_bn(bn_layer)
        return {
            "scale": add_tensor(f"{prefix}.scale", scale),
            "shift": add_tensor(f"{prefix}.shift", shift),
        }

    # Stem: denses[0] + bns[0]
    manifest: dict[str, Any] = {
        "format": "wl-e2e-conformer-v1",
        "featureMode": feature_mode,
        "seqLen": int(model.input_shape[1]),
        "inDim": int(model.input_shape[2]),
        "dim": dim,
        "numHeads": int(mhas[0].num_heads),
        "numClasses": len(classes),
        "lnEps": float(lns[0].epsilon),
        "stem": {
            "proj": add_dense("stem.proj", denses[0]),
            "bn": add_bn("stem.bn", bns[0]),
        },
        "blocks": [],
    }

    dense_idx = 1
    bn_idx = 1
    ln_idx = 0
    for blk_i in range(num_blocks):
        p = f"b{blk_i}"
        ln1 = add_ln(f"{p}.ln1", lns[ln_idx])
        conv_in = add_dense(f"{p}.convIn", denses[dense_idx])
        dw_w, dw_b = [np.asarray(x, dtype=np.float32) for x in dwconvs[blk_i].get_weights()]
        dw = {
            "w": add_tensor(f"{p}.dw.w", dw_w.reshape(dw_w.shape[0], dw_w.shape[1])),
            "b": add_tensor(f"{p}.dw.b", dw_b),
        }
        conv_bn = add_bn(f"{p}.convBn", bns[bn_idx])
        conv_out = add_dense(f"{p}.convOut", denses[dense_idx + 1])

        ln_att = add_ln(f"{p}.lnAtt", lns[ln_idx + 1])
        mha_layer = mhas[blk_i]
        qw, qb, kw, kb, vw, vb, ow, ob = [
            np.asarray(x, dtype=np.float32) for x in mha_layer.get_weights()
        ]
        mha = {
            "qW": add_tensor(f"{p}.mha.qW", qw.reshape(dim, dim).T),
            "qB": add_tensor(f"{p}.mha.qB", qb.reshape(dim)),
            "kW": add_tensor(f"{p}.mha.kW", kw.reshape(dim, dim).T),
            "kB": add_tensor(f"{p}.mha.kB", kb.reshape(dim)),
            "vW": add_tensor(f"{p}.mha.vW", vw.reshape(dim, dim).T),
            "vB": add_tensor(f"{p}.mha.vB", vb.reshape(dim)),
            "oW": add_tensor(f"{p}.mha.oW", ow.reshape(dim, dim).T),
            "oB": add_tensor(f"{p}.mha.oB", ob.reshape(dim)),
        }

        ln_ff = add_ln(f"{p}.lnFf", lns[ln_idx + 2])
        ff1 = add_dense(f"{p}.ff1", denses[dense_idx + 2])
        ff2 = add_dense(f"{p}.ff2", denses[dense_idx + 3])

        manifest["blocks"].append(
            {
                "ln1": ln1,
                "convIn": conv_in,
                "dw": dw,
                "convBn": conv_bn,
                "convOut": conv_out,
                "lnAtt": ln_att,
                "mha": mha,
                "lnFf": ln_ff,
                "ff1": ff1,
                "ff2": ff2,
            }
        )
        dense_idx += 4
        bn_idx += 1
        ln_idx += 3

    manifest["head"] = {
        "lnOut": add_ln("head.lnOut", lns[ln_idx]),
        "dense": add_dense("head.dense", denses[dense_idx]),
    }
    manifest["tensors"] = tensors

    (out_dir / "wl_e2e_weights.bin").write_bytes(b"".join(chunks))
    (out_dir / "wl_e2e_model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_wl_labels_json(out_dir / "bisindo_wl_labels.json", classes)
    return out_dir / "wl_e2e_model.json"


def main() -> int:
    npz_path = Path("data/processed/wl_custom_landmarks.npz")
    pub_dir = Path("public/models/bisindo")
    X_all, y_str, g_all = load_wl_sequences_npz(npz_path)
    is_clean = np.array(["_aug" not in str(grp) for grp in g_all])
    classes = sorted(set(y_str.tolist()))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([label_to_idx[str(s)] for s in y_str], dtype=np.int64)

    # Variant B: End-to-End Conformer (300-dim)
    print("Training Variant B: End-to-End Conformer (300-dim)...")
    model_b = train_e2e_with_mixup(X_all, y_idx, len(classes), epochs=75, seed=42)
    probs_b = model_b.predict(X_all[is_clean], verbose=0)
    acc_b = float(np.mean(np.argmax(probs_b, axis=1) == y_idx[is_clean]))
    print(f"Variant B clean accuracy: {acc_b:.2%}")
    export_e2e_conformer_to_browser(
        model_b,
        classes,
        pub_dir / "variants" / "e2e-conformer",
        feature_mode="standard300",
    )

    # Variant C: End-to-End Geometric Conformer (350-dim)
    print("Training Variant C: End-to-End Geometric Conformer (350-dim)...")
    X_geo = extract_geometric_features(X_all)
    model_c = train_e2e_with_mixup(X_geo, y_idx, len(classes), epochs=75, seed=142)
    probs_c = model_c.predict(X_geo[is_clean], verbose=0)
    acc_c = float(np.mean(np.argmax(probs_c, axis=1) == y_idx[is_clean]))
    print(f"Variant C clean accuracy: {acc_c:.2%}")
    export_e2e_conformer_to_browser(
        model_c,
        classes,
        pub_dir / "variants" / "e2e-geo-conformer",
        feature_mode="geo350",
    )

    variants_manifest = {
        "defaultVariant": "e2e-conformer",
        "variants": [
            {
                "id": "frozen-head",
                "label": "Model A: Frozen Conformer + Head (110 Klip)",
                "basePath": "/models/bisindo",
                "format": "legacy-tflite-conformer",
                "featureMode": "standard300",
            },
            {
                "id": "e2e-conformer",
                "label": "Model B: End-to-End Conformer + MixUp",
                "basePath": "/models/bisindo/variants/e2e-conformer",
                "format": "e2e-conformer",
                "featureMode": "standard300",
            },
            {
                "id": "e2e-geo-conformer",
                "label": "Model C: Geometric Conformer (Jarak+Sudut)",
                "basePath": "/models/bisindo/variants/e2e-geo-conformer",
                "format": "e2e-conformer",
                "featureMode": "geo350",
            },
        ],
    }
    (pub_dir / "wl_variants.json").write_text(
        json.dumps(variants_manifest, indent=2), encoding="utf-8"
    )
    print(f"Exported A/B/C switcher registry -> {pub_dir / 'wl_variants.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
