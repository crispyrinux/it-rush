"""Train the V2 222-dim aspect/distance-invariant BISINDO Alphabet model."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from bisindo.train import run_training, run_v2_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V2 222-dim BISINDO Alphabet MLP -> TFLite + TF.js (with 2-hand focus)"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument(
        "--extra-raw-npz",
        type=Path,
        nargs="*",
        default=[],
        help="Optional extra raw_hands NPZ files to merge",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--aug-copies", type=int, default=3)
    parser.add_argument(
        "--also-refresh-legacy-86",
        action="store_true",
        help="Also retrain artifacts/bisindo_alphabet_model_f32.tflite on enhanced public_landmarks.npz",
    )
    parser.add_argument(
        "--no-sync-public",
        action="store_true",
        help="Do not copy V2 TF.js + TFLite files into public/models/bisindo/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = cfg["paths"]
    t_cfg = cfg["training"]
    aug_cfg = t_cfg["augment"]

    raw_candidates = [
        Path(paths.get("raw_hands_npz", "data/processed/public_raw_hands.npz")),
        *args.extra_raw_npz,
    ]
    existing_raw = [p for p in raw_candidates if p.exists()]
    if not existing_raw:
        raise FileNotFoundError(
            "No raw_hands NPZ files found. Run `python scripts/prepare_dataset.py` first."
        )

    classes = list(cfg["classes"])
    public_dir = None if args.no_sync_public else paths.get("public_models_dir")
    v2_tflite = Path(paths.get("v2_tflite_model", "artifacts/bisindo_alphabet_v2_f32.tflite"))
    v2_labels = Path(paths.get("v2_labels_json", "artifacts/bisindo_alphabet_v2_labels.json"))

    print(f"Training V2 222-dim model from: {[str(p) for p in existing_raw]}")
    result = run_v2_training(
        raw_hands_npz_paths=existing_raw,
        classes=classes,
        tflite_out=v2_tflite,
        labels_out=v2_labels,
        tfjs_dir=paths.get("tfjs_dir"),
        public_models_dir=public_dir,
        seed=t_cfg["seed"],
        val_split=t_cfg["val_split"],
        test_split=t_cfg["test_split"],
        batch_size=t_cfg["batch_size"],
        epochs=args.epochs if args.epochs is not None else t_cfg["epochs"],
        learning_rate=t_cfg["learning_rate"],
        early_stopping_patience=t_cfg["early_stopping_patience"],
        reduce_lr_patience=t_cfg["reduce_lr_patience"],
        aug_copies=args.aug_copies,
        single_hand_ratio=aug_cfg["single_hand_ratio"],
    )

    print(f"\nV2 Test Loss: {result.test_loss:.4f} | Test Accuracy: {result.test_accuracy:.4%}")
    print("\n" + result.report)
    print(f"Exported V2 TFLite : {result.tflite_path}")
    print(f"Exported V2 Labels : {result.labels_path}")
    if result.tfjs_model_path:
        print(f"Exported V2 TF.js  : {result.tfjs_model_path}")
    if public_dir:
        print(f"Synced Frontend V2 : {public_dir}")

    if args.also_refresh_legacy_86:
        legacy_npzs = [
            p for p in [Path(paths["public_npz"]), Path(paths["webcam_npz"])] if p.exists()
        ]
        if legacy_npzs:
            print(f"\nRefreshing legacy 86-dim artifact from: {[str(p) for p in legacy_npzs]}")
            res86 = run_training(
                npz_paths=legacy_npzs,
                classes=classes,
                tflite_out=paths["tflite_model"],
                labels_out=paths["labels_json"],
                tfjs_dir=None,
                seed=t_cfg["seed"],
                val_split=t_cfg["val_split"],
                test_split=t_cfg["test_split"],
                batch_size=t_cfg["batch_size"],
                epochs=args.epochs if args.epochs is not None else t_cfg["epochs"],
                learning_rate=t_cfg["learning_rate"],
                early_stopping_patience=t_cfg["early_stopping_patience"],
                reduce_lr_patience=t_cfg["reduce_lr_patience"],
                flip_swap=aug_cfg["flip_swap"],
                single_hand_ratio=aug_cfg["single_hand_ratio"],
                jitter_std=aug_cfg["jitter_std"],
                jitter_copies=aug_cfg["jitter_copies"],
            )
            print(
                f"Legacy 86-dim Test Accuracy: {res86.test_accuracy:.4%} -> {res86.tflite_path}"
            )


if __name__ == "__main__":
    main()
