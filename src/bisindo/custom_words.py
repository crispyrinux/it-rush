"""Discovery, normalization, metadata management, and synchronization for custom BISINDO words."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Sequence

from bisindo.dataset import VIDEO_EXTS
from bisindo.word_level import WL_RAW_FEATURE_DIM, WL_SEQUENCE_LEN, WL_TEMPORAL_FEATURE_DIM

_VALID_WORD_ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_STALE_VIDEO_EXTS = {".mp4", ".webm", ".webp"}


@dataclass(frozen=True)
class CustomWordEntry:
    """Normalized custom BISINDO word entry discovered from an input folder."""

    word_id: str
    title: str
    video_paths: tuple[Path, ...]
    representative_video: Path
    hands_override: int | None = None
    guide_override: str | None = None


def normalize_word_id(raw_name: str, *, is_filename: bool = False) -> str:
    """Normalize a filename or folder name into a canonical snake_case `word_id`.

    Examples:
        `pantai.mp4` -> `pantai`
        `terima_kasih.mp4` -> `terima_kasih`
        `Terima Kasih__take2.mp4` -> `terima_kasih`
    """
    base = Path(raw_name).stem if is_filename else raw_name
    if is_filename and "__" in base:
        base = base.split("__", 1)[0]
    cleaned = _NON_ALNUM_RE.sub("_", base.strip().lower()).strip("_")
    if not cleaned or not _VALID_WORD_ID_RE.match(cleaned):
        raise ValueError(
            f"Cannot derive a valid alphanumeric word_id from {raw_name!r} (got {cleaned!r})"
        )
    return cleaned


def format_word_title(word_id: str) -> str:
    """Convert a snake_case `word_id` into Title Case for UI display."""
    parts = [part for part in word_id.strip().split("_") if part]
    return " ".join(part.capitalize() for part in parts)


def load_words_meta(input_dir: Path) -> dict[str, dict[str, Any]]:
    """Read optional `words_meta.json` overrides from `input_dir`."""
    meta_path = Path(input_dir) / "words_meta.json"
    if not meta_path.is_file():
        return {}
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {meta_path}, got {type(raw).__name__}")

    normalized: dict[str, dict[str, Any]] = {}
    for raw_key, val in raw.items():
        if not isinstance(val, dict):
            raise ValueError(f"Expected metadata object for key {raw_key!r} in {meta_path}")
        wid = normalize_word_id(str(raw_key))
        entry: dict[str, Any] = {}
        if "title" in val and val["title"] is not None:
            entry["title"] = str(val["title"]).strip()
        if "hands" in val and val["hands"] is not None:
            hands_val = int(val["hands"])
            if hands_val not in (1, 2):
                raise ValueError(f"Invalid hands={hands_val} for {wid!r}; expected 1 or 2")
            entry["hands"] = hands_val
        if "guide" in val and val["guide"] is not None:
            entry["guide"] = str(val["guide"]).strip()
        if "order" in val and val["order"] is not None:
            entry["order"] = int(val["order"])
        normalized[wid] = entry
    return normalized


def _select_representative_video(word_id: str, clips: Sequence[Path]) -> Path:
    exact = [p for p in clips if normalize_word_id(p.name, is_filename=True) == word_id and "__" not in p.stem]
    if exact:
        exact_name = [p for p in exact if p.stem.lower() == word_id]
        return exact_name[0] if exact_name else exact[0]
    return max(clips, key=lambda p: (p.stat().st_size, p.name))


def discover_custom_word_videos(input_dir: str | Path) -> list[CustomWordEntry]:
    """Scan `input_dir` for flat videos and/or word subfolders and group them by `word_id`."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    meta = load_words_meta(root)
    grouped: dict[str, list[Path]] = {}
    seen_paths: dict[str, set[Path]] = {}

    def _add_clip(wid: str, clip_path: Path) -> None:
        resolved = clip_path.resolve()
        bucket = grouped.setdefault(wid, [])
        seen = seen_paths.setdefault(wid, set())
        if resolved not in seen:
            seen.add(resolved)
            bucket.append(clip_path)

    for child in sorted(root.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_file() and child.suffix.lower() in VIDEO_EXTS:
            wid = normalize_word_id(child.name, is_filename=True)
            _add_clip(wid, child)
        elif child.is_dir():
            clips = sorted(
                p for p in child.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
            )
            if not clips:
                continue
            wid = normalize_word_id(child.name, is_filename=False)
            for clip in clips:
                _add_clip(wid, clip)

    if len(grouped) < 2:
        raise ValueError(
            f"Minimal 2 kata (video .mp4) dibutuhkan di {root}, ditemukan {len(grouped)}: {sorted(grouped)}"
        )

    entries: list[tuple[tuple[int, str], CustomWordEntry]] = []
    for wid in sorted(grouped):
        clips = sorted(grouped[wid], key=lambda p: (p.name.lower(), str(p)))
        rep = _select_representative_video(wid, clips)
        m = meta.get(wid, {})
        order = int(m.get("order", 10_000))
        title = str(m["title"]) if m.get("title") else format_word_title(wid)
        hands_ovr = int(m["hands"]) if "hands" in m else None
        guide_ovr = str(m["guide"]) if m.get("guide") else None
        entries.append(
            (
                (order, wid),
                CustomWordEntry(
                    word_id=wid,
                    title=title,
                    video_paths=tuple(clips),
                    representative_video=rep,
                    hands_override=hands_ovr,
                    guide_override=guide_ovr,
                ),
            )
        )

    entries.sort(key=lambda item: item[0])
    return [item[1] for item in entries]


def infer_hands_count(sim_frames: Sequence[dict[str, Any]]) -> int:
    """Infer whether a word uses 1 or 2 hands from extracted 30-frame skeleton detections."""
    hand_frames = 0
    two_hand_frames = 0
    for frame in sim_frames:
        n_hands = len(frame.get("hands") or ())
        if n_hands >= 1:
            hand_frames += 1
        if n_hands >= 2:
            two_hand_frames += 1

    if hand_frames == 0:
        return 2
    return 2 if (two_hand_frames / hand_frames) >= 0.35 else 1


def build_default_guide(title: str, hands: int) -> str:
    """Build a clear Indonesian instruction string for a newly added BISINDO word."""
    hand_text = "kedua tangan" if hands == 2 else "satu tangan"
    return (
        f'Amati animasi contoh kata "{title}", lalu praktikkan gerakan dengan {hand_text} '
        "di depan kamera selama 2,5 detik."
    )


def sync_directory_mp4s(target_dir: Path, keep_filenames: set[str]) -> list[Path]:
    """Remove stale encoded video files inside `target_dir` not present in `keep_filenames`."""
    out_dir = Path(target_dir)
    if not out_dir.is_dir():
        return []
    removed: list[Path] = []
    for child in sorted(out_dir.iterdir()):
        if not child.is_file():
            continue
        if child.suffix.lower() in _STALE_VIDEO_EXTS and child.name not in keep_filenames:
            child.unlink()
            removed.append(child)
    return removed


def write_wl_labels_json(path: str | Path, classes: Sequence[str]) -> Path:
    """Write `bisindo_wl_labels.json` matching the WL-BISINDO browser & Python schema."""
    class_list = [str(c) for c in classes]
    payload = {
        "label_to_index": {lbl: idx for idx, lbl in enumerate(class_list)},
        "index_to_label": {str(idx): lbl for idx, lbl in enumerate(class_list)},
        "num_classes": len(class_list),
        "labels": class_list,
        "feature_dim": WL_RAW_FEATURE_DIM,
        "sequence_len": WL_SEQUENCE_LEN,
        "dataset": "WL-BISINDO",
        "normalization": "nose-centered + per-sequence std",
        "temporal_total": WL_TEMPORAL_FEATURE_DIM,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def write_frontend_words_json(
    path: str | Path,
    entries: Sequence[dict[str, Any]],
    model_version: str,
) -> Path:
    """Write single-source-of-truth `lib/bisindoWords.json` for the Next.js app."""
    payload = {
        "modelVersion": str(model_version),
        "words": [
            {
                "id": str(item["id"]),
                "title": str(item["title"]),
                "hands": 1 if int(item["hands"]) == 1 else 2,
                "guide": str(item["guide"]),
                "localVideoPath": str(item["localVideoPath"]),
            }
            for item in entries
        ],
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path
