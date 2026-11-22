"""Unit tests for the 2.5-second training clip recorder helpers."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.record_word_clips import (
    count_existing_clips,
    load_word_choices,
    next_clip_path,
    save_frames_as_mp4,
)


def test_load_word_choices_and_clip_numbering(tmp_path: Path):
    words_json = tmp_path / "bisindoWords.json"
    words_json.write_text(
        json.dumps(
            {
                "modelVersion": "v1",
                "words": [
                    {"id": "alam", "title": "Alam"},
                    {"id": "iklim", "title": "Iklim"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert load_word_choices(words_json) == [("alam", "Alam"), ("iklim", "Iklim")]

    word_dir = tmp_path / "alam"
    assert count_existing_clips(word_dir) == 0
    first = next_clip_path(word_dir)
    assert first == word_dir / "take_001.mp4"

    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(10)]
    save_frames_as_mp4(frames, first, fps=20.0)
    assert first.is_file()
    assert count_existing_clips(word_dir) == 1
    assert next_clip_path(word_dir) == word_dir / "take_002.mp4"

    cap = cv2.VideoCapture(str(first))
    try:
        assert cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count == 10
    finally:
        cap.release()
