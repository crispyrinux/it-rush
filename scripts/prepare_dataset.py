"""Extract BISINDO hand landmarks (raw 2-hand coords + 86-dim features) from image folders."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from bisindo.dataset import (
    append_landmarks_npz,
    extract_raw_hands_from_directory,
    load_raw_hands_npz,
    save_landmarks_npz,
    save_raw_hands_npz,
)
from bisindo.features import (
    FEATURE_DIM,
    TWO_HAND_LETTERS,
    build_single_hand_features,
    build_two_hand_features,
)
from bisindo.landmarks import create_hands_detector


def _find_yolo_label_file(img_path: Path, input_dir: Path) -> Path | None:
    same_dir = img_path.with_suffix(".txt")
    if same_dir.exists():
        return same_dir

    rel = img_path.relative_to(input_dir)
    candidates = [
        input_dir.parent / "labels" / rel.with_suffix(".txt"),
        input_dir / "labels" / rel.with_suffix(".txt"),
        img_path.parent.parent / "labels" / img_path.parent.name / f"{img_path.stem}.txt",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _build_legacy_86_from_raw(hands: np.ndarray, num_hands: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    for i in range(len(hands)):
        if int(num_hands[i]) >= 2:
            rows.append(build_two_hand_features(hands[i, 0], hands[i, 1]))
        else:
            rows.append(build_single_hand_features(hands[i, 0]))
    if not rows:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare BISINDO raw_hands + landmark NPZ from image folders"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--input-dir", type=Path, default=None, help="Override raw_images_dir")
    parser.add_argument("--output", type=Path, default=None, help="Override public_npz (86-dim)")
    parser.add_argument(
        "--raw-hands-output",
        type=Path,
        default=None,
        help="Override raw_hands_npz (raw 2x21x2 landmarks + aspect ratio)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.15,
        help="Minimum MediaPipe hand detection/tracking confidence for extraction",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append extracted samples to existing NPZ files instead of overwriting",
    )
    parser.add_argument(
        "--allow-new-classes",
        action="store_true",
        help="Extract all subfolder names as classes, including classes not in config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    classes = set(cfg["classes"])
    input_dir = args.input_dir or Path(cfg["paths"]["raw_images_dir"])
    output_path = args.output or Path(cfg["paths"]["public_npz"])
    raw_hands_output = args.raw_hands_output or Path(
        cfg["paths"].get("raw_hands_npz", "data/processed/public_raw_hands.npz")
    )

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    detector = create_hands_detector(
        static_image_mode=True,
        min_detection_confidence=float(args.min_confidence),
        min_tracking_confidence=float(args.min_confidence),
    )
    try:
        hands, num_hands, aspect_ratios, y, groups, discovered = extract_raw_hands_from_directory(
            input_dir=input_dir,
            classes=classes,
            detector=detector,
            two_hand_classes=TWO_HAND_LETTERS,
            allow_new_classes=args.allow_new_classes,
            find_yolo_fn=_find_yolo_label_file,
        )
    finally:
        detector.close()

    if len(hands) == 0:
        raise RuntimeError("No hand landmarks extracted from any image.")

    counts = Counter(y.tolist())
    two_tot = 0
    two_det2 = 0
    print("=== Extraction Summary ===")
    for label in discovered:
        mask = y == label
        n_cls = int(counts[label])
        n_2h = int(np.sum(num_hands[mask] >= 2))
        extra = f" (2-hand detected: {n_2h}/{n_cls} = {n_2h / n_cls:.1%})" if label in TWO_HAND_LETTERS else ""
        print(f"[{label}] {n_cls} samples{extra}")
        if label in TWO_HAND_LETTERS:
            two_tot += n_cls
            two_det2 += n_2h

    if two_tot > 0:
        print(
            f"Overall 2-hand detection across 2H letters: {two_det2}/{two_tot} ({two_det2 / two_tot:.1%})"
        )

    X_86 = _build_legacy_86_from_raw(hands, num_hands)
    if args.append:
        total_86 = append_landmarks_npz(output_path, X_86, y, groups=groups)
        print(f"Appended {len(X_86)} 86-dim samples to {output_path} (total={total_86})")
        if raw_hands_output.exists():
            h_old, nh_old, ar_old, y_old, g_old = load_raw_hands_npz(raw_hands_output)
            if g_old is None:
                g_old = np.array([f"old_{i}" for i in range(len(h_old))], dtype=str)
            save_raw_hands_npz(
                raw_hands_output,
                np.concatenate([h_old, hands], axis=0),
                np.concatenate([nh_old, num_hands], axis=0),
                np.concatenate([ar_old, aspect_ratios], axis=0),
                np.concatenate([y_old, y], axis=0),
                groups=np.concatenate([g_old, groups], axis=0),
            )
        else:
            save_raw_hands_npz(raw_hands_output, hands, num_hands, aspect_ratios, y, groups=groups)
    else:
        save_landmarks_npz(output_path, X_86, y, groups=groups)
        save_raw_hands_npz(raw_hands_output, hands, num_hands, aspect_ratios, y, groups=groups)
        print(f"Saved {len(X_86)} samples -> {output_path} and {raw_hands_output}")


if __name__ == "__main__":
    main()
