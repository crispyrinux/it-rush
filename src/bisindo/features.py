"""Bohacek hand landmark normalization and 86-dim BISINDO feature builder.

Matches KawanIsyarat's `BisindoAlphabetService` feature layout:
  [0:42]   Leftmost / first hand (21 x (x, y) Bohacek-normalized to [0, 1])
  [42:84]  Rightmost / second hand (21 x (x, y) Bohacek-normalized, or zeros)
  [84]     dx = mean_x(right) - mean_x(left) (always >= 0 for two hands, 0.0 for single)
  [85]     dy = mean_y(right) - mean_y(left) (0.0 for single)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

NUM_LANDMARKS = 21
NUM_HAND_FEATURES = NUM_LANDMARKS * 2  # 42
FEATURE_DIM = NUM_HAND_FEATURES * 2 + 2  # 86
INTER_HAND_KEY_INDICES: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 9)
FEATURE_DIM_V2 = (
    NUM_HAND_FEATURES * 4 + len(INTER_HAND_KEY_INDICES) * len(INTER_HAND_KEY_INDICES) + 5
)  # 222
TWO_HAND_LETTERS: frozenset[str] = frozenset(
    {"A", "B", "D", "F", "G", "H", "K", "M", "N", "P", "Q", "S", "T", "W", "X", "Y"}
)
_EPS = 1e-6


def _validate_hand_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float32)
    if arr.shape == (NUM_HAND_FEATURES,):
        arr = arr.reshape(NUM_LANDMARKS, 2)
    elif arr.shape != (NUM_LANDMARKS, 2):
        raise ValueError(
            f"Expected hand coords shape ({NUM_LANDMARKS}, 2) or ({NUM_HAND_FEATURES},), got {arr.shape}"
        )
    return arr


def normalize_hand_bohacek(coords: np.ndarray) -> np.ndarray:
    """Apply Bohacek bounding-box normalization to 21 (x, y) landmarks.

    Returns a new flat float32 array of length 42 with values in [0, 1].
    """
    pts = _validate_hand_coords(coords)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = np.maximum(np.abs(max_xy - min_xy), _EPS)

    norm = (pts - min_xy) / span
    return norm.astype(np.float32).reshape(NUM_HAND_FEATURES)


def build_two_hand_features(hand_a: np.ndarray, hand_b: np.ndarray) -> np.ndarray:
    """Build 86-dim feature vector from two hands, sorted left-to-right by mean x."""
    pts_a = _validate_hand_coords(hand_a)
    pts_b = _validate_hand_coords(hand_b)

    cx_a = float(pts_a[:, 0].mean())
    cx_b = float(pts_b[:, 0].mean())
    left_pts, right_pts = (pts_a, pts_b) if cx_a <= cx_b else (pts_b, pts_a)

    left_norm = normalize_hand_bohacek(left_pts)
    right_norm = normalize_hand_bohacek(right_pts)

    dx = np.float32(right_pts[:, 0].mean() - left_pts[:, 0].mean())
    dy = np.float32(right_pts[:, 1].mean() - left_pts[:, 1].mean())

    return np.concatenate([left_norm, right_norm, np.array([dx, dy], dtype=np.float32)])


def build_single_hand_features(hand: np.ndarray) -> np.ndarray:
    """Build 86-dim feature vector from a single hand (zero-pads second slot)."""
    pts = _validate_hand_coords(hand)
    hand_norm = normalize_hand_bohacek(pts)
    zeros = np.zeros(NUM_HAND_FEATURES + 2, dtype=np.float32)
    return np.concatenate([hand_norm, zeros])


def _is_second_hand_present(features: np.ndarray) -> bool:
    return bool(np.any(np.abs(features[NUM_HAND_FEATURES : NUM_HAND_FEATURES * 2]) > _EPS))


def flip_and_swap_features(features: np.ndarray) -> np.ndarray:
    """Horizontally flip x-coordinates (x -> 1 - x) and swap hand slots.

    Preserves the invariant that slot 1 is the leftmost hand (dx >= 0).
    Returns a new array without mutating `features`.
    """
    feat = np.asarray(features, dtype=np.float32).copy()
    if feat.shape != (FEATURE_DIM,):
        raise ValueError(f"Expected feature shape ({FEATURE_DIM},), got {feat.shape}")

    if not _is_second_hand_present(feat):
        feat[0:NUM_HAND_FEATURES:2] = 1.0 - feat[0:NUM_HAND_FEATURES:2]
        return feat

    left = feat[0:NUM_HAND_FEATURES].copy()
    right = feat[NUM_HAND_FEATURES : NUM_HAND_FEATURES * 2].copy()

    left[0::2] = 1.0 - left[0::2]
    right[0::2] = 1.0 - right[0::2]

    dx = feat[84]
    dy = -feat[85]
    return np.concatenate([right, left, np.array([dx, dy], dtype=np.float32)])


def make_single_hand_fallback(features: np.ndarray, keep: str = "left") -> np.ndarray:
    """Drop one hand slot and zero-pad slot 2 to simulate overlapping-hand occlusion."""
    feat = np.asarray(features, dtype=np.float32)
    if feat.shape != (FEATURE_DIM,):
        raise ValueError(f"Expected feature shape ({FEATURE_DIM},), got {feat.shape}")

    if keep == "left":
        kept = feat[0:NUM_HAND_FEATURES].copy()
    elif keep == "right":
        kept = feat[NUM_HAND_FEATURES : NUM_HAND_FEATURES * 2].copy()
    else:
        raise ValueError(f"keep must be 'left' or 'right', got {keep!r}")

    zeros = np.zeros(NUM_HAND_FEATURES + 2, dtype=np.float32)
    return np.concatenate([kept, zeros])


def add_landmark_jitter(
    features: np.ndarray,
    std: float = 0.015,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian jitter to active hand landmark slots and clip to [0, 1].

    Returns a new array without mutating `features`.
    """
    if rng is None:
        rng = np.random.default_rng()

    out = np.asarray(features, dtype=np.float32).copy()
    if out.shape != (FEATURE_DIM,):
        raise ValueError(f"Expected feature shape ({FEATURE_DIM},), got {out.shape}")

    noise1 = rng.normal(0.0, std, size=NUM_HAND_FEATURES).astype(np.float32)
    out[0:NUM_HAND_FEATURES] = np.clip(out[0:NUM_HAND_FEATURES] + noise1, 0.0, 1.0)

    if _is_second_hand_present(out):
        noise2 = rng.normal(0.0, std, size=NUM_HAND_FEATURES).astype(np.float32)
        out[NUM_HAND_FEATURES : NUM_HAND_FEATURES * 2] = np.clip(
            out[NUM_HAND_FEATURES : NUM_HAND_FEATURES * 2] + noise2, 0.0, 1.0
        )
        out[84] = np.float32(max(0.0, out[84] + float(rng.normal(0.0, std * 0.5))))
        out[85] = np.float32(out[85] + float(rng.normal(0.0, std * 0.5)))

    return out


def to_square_coords(coords: np.ndarray, aspect_ratio: float = 1.0) -> np.ndarray:
    """Convert normalized image-space `(21, 2)` coordinates to aspect-corrected square space."""
    pts = _validate_hand_coords(coords).copy()
    pts[:, 0] *= np.float32(aspect_ratio)
    return pts


def normalize_hand_isotropic(pts_sq: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Center a hand around its bounding-box midpoint with uniform span scaling.

    Returns `(normalized_pts_21x2, span, mean_xy)` without distorting finger aspect ratio.
    """
    pts = _validate_hand_coords(pts_sq)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1], _EPS))
    center = (min_xy + max_xy) * np.float32(0.5)
    norm = ((pts - center) / np.float32(span)).astype(np.float32)
    return norm, span, pts.mean(axis=0).astype(np.float32)


def build_bisindo_v2_features(
    hands_xy: Sequence[np.ndarray],
    aspect_ratio: float = 1.0,
    swap_slots: bool = False,
) -> np.ndarray:
    """Build 222-dim aspect/distance-invariant features for 1-hand and 2-hand BISINDO signs.

    Layout:
      [0:42]    Leftmost hand isotropic normalized coords
      [42:84]   Rightmost hand isotropic normalized coords (zeros if 1 hand)
      [84:126]  Leftmost hand in joint two-hand square bounding frame
      [126:168] Rightmost hand in joint two-hand square bounding frame (zeros if 1 hand)
      [168:217] 7x7 normalized inter-hand pairwise key-landmark distances (zeros if 1 hand)
      [217:222] [is_two_hands, dx_rel, dy_rel, span1/bspan, span2/bspan]
    """
    if len(hands_xy) == 0:
        raise ValueError("hands_xy must contain at least one hand")

    sq_hands = [to_square_coords(h, aspect_ratio=aspect_ratio) for h in hands_xy[:2]]
    if len(sq_hands) == 1:
        iso1, _, _ = normalize_hand_isotropic(sq_hands[0])
        iso1_flat = iso1.reshape(NUM_HAND_FEATURES)
        zeros_hand = np.zeros(NUM_HAND_FEATURES, dtype=np.float32)
        zeros_inter = np.zeros(len(INTER_HAND_KEY_INDICES) ** 2, dtype=np.float32)
        meta = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return np.concatenate([iso1_flat, zeros_hand, iso1_flat, zeros_hand, zeros_inter, meta]).astype(
            np.float32
        )

    c0_x = float(sq_hands[0][:, 0].mean())
    c1_x = float(sq_hands[1][:, 0].mean())
    first_is_left = (c0_x <= c1_x) ^ bool(swap_slots)
    left_sq, right_sq = (sq_hands[0], sq_hands[1]) if first_is_left else (sq_hands[1], sq_hands[0])

    iso1, span1, m1 = normalize_hand_isotropic(left_sq)
    iso2, span2, m2 = normalize_hand_isotropic(right_sq)

    both = np.concatenate([left_sq, right_sq], axis=0)
    bmin = both.min(axis=0)
    bmax = both.max(axis=0)
    bspan = float(max(bmax[0] - bmin[0], bmax[1] - bmin[1], _EPS))
    bctr = (bmin + bmax) * np.float32(0.5)

    joint1 = ((left_sq - bctr) / np.float32(bspan)).astype(np.float32).reshape(NUM_HAND_FEATURES)
    joint2 = ((right_sq - bctr) / np.float32(bspan)).astype(np.float32).reshape(NUM_HAND_FEATURES)

    k1 = left_sq[list(INTER_HAND_KEY_INDICES)]
    k2 = right_sq[list(INTER_HAND_KEY_INDICES)]
    diff = k1[:, None, :] - k2[None, :, :]
    dists = (np.sqrt(np.sum(diff * diff, axis=-1)) / np.float32(bspan)).astype(np.float32).reshape(-1)

    dx_rel = np.float32((m2[0] - m1[0]) / bspan)
    dy_rel = np.float32((m2[1] - m1[1]) / bspan)
    meta = np.array([1.0, dx_rel, dy_rel, span1 / bspan, span2 / bspan], dtype=np.float32)

    return np.concatenate(
        [
            iso1.reshape(NUM_HAND_FEATURES),
            iso2.reshape(NUM_HAND_FEATURES),
            joint1,
            joint2,
            dists,
            meta,
        ]
    ).astype(np.float32)


def transform_hands_raw(
    hands_xy: Sequence[np.ndarray],
    aspect_ratio: float = 1.0,
    *,
    flip: bool = False,
    rot_deg: float = 0.0,
    scale_dist: float = 1.0,
    aspect_mult: float = 1.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], float]:
    """Apply realistic camera-space geometric augmentation without mutating input hands."""
    new_ar = float(aspect_ratio) * float(aspect_mult)
    if len(hands_xy) == 0:
        return [], new_ar

    if rng is None:
        rng = np.random.default_rng()

    sq_hands = [to_square_coords(h, aspect_ratio=aspect_ratio) for h in hands_xy]
    all_pts = np.concatenate(sq_hands, axis=0)
    ctr = all_pts.mean(axis=0)

    rad = np.deg2rad(float(rot_deg))
    ca, sa = float(np.cos(rad)), float(np.sin(rad))
    rot_mat = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
    center_target = np.array([0.5 * new_ar, 0.5], dtype=np.float32)

    transformed: list[np.ndarray] = []
    for hand_sq in sq_hands:
        pts = hand_sq - ctr
        if flip:
            pts[:, 0] = -pts[:, 0]
        pts = pts @ rot_mat.T
        pts = pts * np.float32(scale_dist) + center_target
        if noise_std > 0.0:
            span = float(
                max(
                    pts[:, 0].max() - pts[:, 0].min(),
                    pts[:, 1].max() - pts[:, 1].min(),
                    1e-4,
                )
            )
            pts = pts + rng.normal(0.0, noise_std * span, size=pts.shape).astype(np.float32)
        pts[:, 0] /= np.float32(new_ar)
        transformed.append(pts.astype(np.float32))

    return transformed, new_ar
