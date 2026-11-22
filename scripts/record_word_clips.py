"""Interactive 2.5-second webcam clip recorder for WL-BISINDO word training.

Controls inside the preview window:
  SPACE       : Start countdown then record a 2.5s MP4 clip
  N / ]       : Next word
  P / [       : Previous word
  D / Backspc : Undo (delete) the last recorded clip for the current word
  Q / ESC     : Quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import cv2
import numpy as np


def load_word_choices(
    words_json_path: str | Path = Path("lib/bisindoWords.json"),
) -> list[tuple[str, str]]:
    """Load `(word_id, title)` pairs from `lib/bisindoWords.json`."""
    path = Path(words_json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Word catalog not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    words = payload.get("words") or []
    if not words:
        raise ValueError(f"No words found in {path}")
    return [(str(w["id"]), str(w["title"])) for w in words]


def count_existing_clips(word_dir: Path) -> int:
    """Count existing `.mp4` clips in `word_dir`."""
    if not word_dir.is_dir():
        return 0
    return sum(1 for p in word_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")


def next_clip_path(word_dir: Path, prefix: str = "take") -> Path:
    """Return the next non-colliding `take_001.mp4` path inside `word_dir`."""
    word_dir.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        candidate = word_dir / f"{prefix}_{idx:03d}.mp4"
        if not candidate.exists():
            return candidate
        idx += 1


def save_frames_as_mp4(
    frames: Sequence[np.ndarray],
    output_path: Path,
    fps: float = 24.0,
) -> Path:
    """Write a sequence of BGR frames to an MP4 file."""
    if not frames:
        raise ValueError("Cannot save an empty frame sequence")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    safe_fps = max(5.0, float(fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, safe_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record repeated 2.5s BISINDO training clips per word into data/training_word_videos/"
    )
    parser.add_argument("--words-json", type=Path, default=Path("lib/bisindoWords.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/training_word_videos"))
    parser.add_argument("--word", default=None, help="Start on a specific word_id (e.g. alam)")
    parser.add_argument("--duration", type=float, default=2.5, help="Clip duration in seconds (default: 2.5)")
    parser.add_argument("--countdown", type=float, default=1.0, help="Pre-record countdown in seconds (default: 1.0)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=24.0)
    return parser.parse_args()


def _draw_hud(
    canvas: np.ndarray,
    *,
    word_idx: int,
    total_words: int,
    word_id: str,
    title: str,
    saved_count: int,
    state_label: str,
    state_color: tuple[int, int, int],
    progress: float | None = None,
    last_message: str = "",
) -> None:
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 92), (24, 24, 48), -1)
    cv2.rectangle(overlay, (0, h - 44), (w, h), (24, 24, 48), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

    cv2.putText(
        canvas,
        f"[{word_idx + 1}/{total_words}] {title} ({word_id})  |  Saved: {saved_count}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        state_label,
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        state_color,
        2,
        cv2.LINE_AA,
    )

    if progress is not None:
        bar_w = max(1, int((w - 32) * min(1.0, max(0.0, progress))))
        cv2.rectangle(canvas, (16, 76), (w - 16, 86), (90, 90, 120), -1)
        cv2.rectangle(canvas, (16, 76), (16 + bar_w, 86), state_color, -1)

    footer = last_message or "SPACE: Rekam 2.5s | N/P: Ganti Kata | D: Hapus Terakhir | Q: Keluar"
    cv2.putText(
        canvas,
        footer,
        (16, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 245),
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()
    choices = load_word_choices(args.words_json)
    word_ids = [wid for wid, _ in choices]

    idx = 0
    if args.word:
        if args.word not in word_ids:
            raise ValueError(f"Unknown word_id {args.word!r}. Available: {word_ids}")
        idx = word_ids.index(args.word)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    state = "idle"  # idle | countdown | recording
    phase_start = 0.0
    recorded_frames: list[np.ndarray] = []
    saved_history: list[Path] = []
    last_message = ""
    window_name = "SIGNIFY - Rekam Video Training 2.5 Detik"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.monotonic()
            word_id, title = choices[idx]
            word_dir = args.out_dir / word_id
            saved_count = count_existing_clips(word_dir)

            progress: float | None = None
            state_label = "SIAP - Tekan SPACE untuk rekam 2.5 detik"
            state_color = (120, 235, 170)

            if state == "countdown":
                elapsed = now - phase_start
                rem = max(0.0, args.countdown - elapsed)
                if rem <= 0.0:
                    state = "recording"
                    phase_start = now
                    recorded_frames = [frame.copy()]
                    state_label = f"MEREKAM... 0.0 / {args.duration:.1f}s"
                    state_color = (80, 90, 255)
                    progress = 0.0
                else:
                    state_label = f"BERSIAP... {rem:.1f}s"
                    state_color = (60, 210, 255)
                    progress = elapsed / max(args.countdown, 0.01)
            elif state == "recording":
                recorded_frames.append(frame.copy())
                elapsed = now - phase_start
                progress = elapsed / max(args.duration, 0.1)
                state_label = f"MEREKAM... {min(elapsed, args.duration):.1f} / {args.duration:.1f}s"
                state_color = (80, 90, 255)
                if elapsed >= args.duration:
                    out_path = next_clip_path(word_dir)
                    actual_fps = len(recorded_frames) / max(elapsed, 0.1)
                    save_frames_as_mp4(recorded_frames, out_path, fps=actual_fps)
                    saved_history.append(out_path)
                    last_message = f"Tersimpan: {word_id}/{out_path.name} ({len(recorded_frames)} frame)"
                    print(f"[SAVED] {out_path} ({len(recorded_frames)} frames @ {actual_fps:.1f} fps)")
                    recorded_frames = []
                    state = "idle"
                    progress = None

            display = cv2.flip(frame, 1)
            _draw_hud(
                display,
                word_idx=idx,
                total_words=len(choices),
                word_id=word_id,
                title=title,
                saved_count=count_existing_clips(word_dir),
                state_label=state_label,
                state_color=state_color,
                progress=progress,
                last_message=last_message,
            )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if state == "idle":
                if key == ord(" "):
                    last_message = ""
                    if args.countdown > 0:
                        state = "countdown"
                        phase_start = time.monotonic()
                    else:
                        state = "recording"
                        phase_start = time.monotonic()
                        recorded_frames = []
                elif key in (ord("n"), ord("N"), ord("]")):
                    idx = (idx + 1) % len(choices)
                    last_message = f"Kata aktif: {choices[idx][1]}"
                elif key in (ord("p"), ord("P"), ord("[")):
                    idx = (idx - 1) % len(choices)
                    last_message = f"Kata aktif: {choices[idx][1]}"
                elif key in (ord("d"), ord("D"), 8):
                    existing = sorted(word_dir.glob("*.mp4")) if word_dir.is_dir() else []
                    if existing:
                        removed = existing[-1]
                        removed.unlink()
                        last_message = f"Dihapus: {word_id}/{removed.name}"
                        print(f"[DELETED] {removed}")
                    else:
                        last_message = f"Belum ada klip untuk {title}"
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
