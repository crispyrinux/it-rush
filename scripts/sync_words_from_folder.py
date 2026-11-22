"""One-command pipeline to replace all BISINDO website words & AI model from a video folder."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import cv2
import numpy as np
import yaml

from bisindo.custom_words import (
    CustomWordEntry,
    build_default_guide,
    discover_custom_word_videos,
    infer_hands_count,
    sync_directory_mp4s,
    write_frontend_words_json,
    write_wl_labels_json,
)
from bisindo.landmarks import create_hands_detector, detect_raw_hands
from bisindo.word_demos import (
    DEFAULT_BUCKET,
    DEFAULT_TABLE,
    WordDemoSpec,
    build_word_demo_row,
    delete_stale_word_demo_rows,
    encode_word_demo,
    load_env_file,
    render_powershell_upload_script,
    upload_file_to_supabase_storage,
    upsert_word_demo_rows,
)
from bisindo.word_level import (
    POSE_ANCHOR_IDX,
    WL_SEQUENCE_LEN,
    augment_wl_raw_frames,
    create_pose_detector,
    detect_raw_pose,
    extract_wl_backbone_embeddings,
    extract_wl_frame_features,
    export_wl_browser_weights,
    predict_wl_with_head,
    prepare_wl_model_input,
    save_wl_sequences_npz,
    train_wl_custom_head,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace all website BISINDO words, demo videos, and AI model from a folder of videos."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/custom_word_videos"),
        help="Folder containing <word>.mp4 files or <word>/*.mp4 subfolders (default: data/custom_word_videos)",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--augment-per-class", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--l2-reg", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-version",
        default=None,
        help="Cache-busting version string (default: custom-YYYYMMDD-HHMMSS)",
    )
    parser.add_argument(
        "--public-videos-dir",
        type=Path,
        default=Path("public/videos/bisindo"),
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("exports/word_demos_bundle"),
    )
    parser.add_argument(
        "--sim-sequences-out",
        type=Path,
        default=Path("public/samples/bisindo/wl_sequences.json"),
    )
    parser.add_argument(
        "--frontend-words-json",
        type=Path,
        default=Path("lib/bisindoWords.json"),
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--upload-supabase", action="store_true")
    parser.add_argument("--skip-video-encode", action="store_true")
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Only update demo videos, PowerShell bundle, and lib/bisindoWords.json without training the AI model",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _extract_visual_frames(
    video_path: Path,
    hand_detector: Any,
    pose_detector: Any,
) -> tuple[list[np.ndarray], list[dict[str, Any]], int, int]:
    """Read video once -> `(raw_100_frames, resampled_30_sim_frames, hand_hits, two_hand_hits)`."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    bgr_frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            bgr_frames.append(frame)
    finally:
        cap.release()

    if not bgr_frames:
        raise ValueError(f"Video file contained no readable frames: {video_path}")

    raw_frames: list[np.ndarray] = []
    all_visual: list[dict[str, Any]] = []
    for frame in bgr_frames:
        raw_hands = detect_raw_hands(frame, hand_detector)
        raw_pose = detect_raw_pose(frame, pose_detector)
        raw_frames.append(extract_wl_frame_features(raw_hands, raw_pose))

        hands_out = [
            [[round(float(lm.x), 4), round(float(lm.y), 4)] for lm in hand_lms[:21]]
            for hand_lms in raw_hands[:2]
        ]
        pose_out = (
            [
                [round(float(raw_pose[idx].x), 4), round(float(raw_pose[idx].y), 4)]
                for idx in POSE_ANCHOR_IDX
            ]
            if raw_pose
            else None
        )
        all_visual.append({"hands": hands_out, "pose": pose_out})

    sim_idx = np.linspace(0, len(all_visual) - 1, WL_SEQUENCE_LEN, dtype=int)
    sim_frames = [all_visual[int(k)] for k in sim_idx]
    hand_hits = sum(1 for f in sim_frames if len(f["hands"]) >= 1)
    two_hand_hits = sum(1 for f in sim_frames if len(f["hands"]) >= 2)
    return raw_frames, sim_frames, hand_hits, two_hand_hits


def _resolve_supabase_credentials(env_file: Path) -> tuple[str, str]:
    file_env = load_env_file(env_file)
    url = (
        os.environ.get("SUPABASE_URL")
        or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        or file_env.get("SUPABASE_URL")
        or file_env.get("NEXT_PUBLIC_SUPABASE_URL")
        or ""
    ).strip()
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or file_env.get("SUPABASE_SERVICE_ROLE_KEY")
        or ""
    ).strip()
    return url, key


def _print_discovery_summary(entries: list[CustomWordEntry]) -> None:
    print(f"Ditemukan {len(entries)} kata baru:")
    for idx, item in enumerate(entries, start=1):
        hands_note = f"{item.hands_override} tangan (meta)" if item.hands_override else "auto-detect"
        print(
            f"  {idx:2d}. {item.word_id:18s} | title='{item.title}' | clips={len(item.video_paths)} | "
            f"rep='{item.representative_video.name}' | hands={hands_note}"
        )


def main() -> int:
    args = parse_args()
    entries = discover_custom_word_videos(args.input_dir)
    _print_discovery_summary(entries)

    if args.dry_run:
        print("\n[DRY-RUN] Tidak ada file yang diubah.")
        return 0

    if args.demo_only and args.skip_video_encode:
        raise ValueError("--demo-only cannot be combined with --skip-video-encode")

    model_version = args.model_version or f"custom-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    classes = [e.word_id for e in entries]
    best_sim_by_word: dict[str, tuple[int, list[dict[str, Any]]]] = {}

    if not args.demo_only:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        paths = cfg["paths"]
        wl_tflite = Path(paths["wl_tflite_model"])
        wl_labels = Path(paths["wl_labels_json"])
        pub_dir = Path(paths["public_models_dir"])
        custom_npz = Path("data/processed/wl_custom_landmarks.npz")

        if not wl_tflite.exists():
            raise FileNotFoundError(f"Backbone WL TFLite model not found: {wl_tflite}")

        hand_det = create_hands_detector(static_image_mode=True, min_detection_confidence=0.3)
        pose_det = create_pose_detector(min_detection_confidence=0.3)

        clean_clips: list[tuple[str, str, list[np.ndarray]]] = []
        try:
            for entry in entries:
                for vid_path in entry.video_paths:
                    raw_frames, sim_frames, hand_hits, two_hits = _extract_visual_frames(
                        vid_path, hand_det, pose_det
                    )
                    if hand_hits == 0:
                        print(
                            f"  [WARN] '{vid_path}' tidak mendeteksi tangan pada 30 frame sampel; "
                            "pastikan pencahayaan terang dan tangan terlihat jelas."
                        )
                    clean_clips.append((entry.word_id, vid_path.name, raw_frames))
                    pose_hits = sum(1 for f in sim_frames if f["pose"] is not None)
                    score = two_hits * 3 + hand_hits * 2 + pose_hits
                    prev_score = best_sim_by_word[entry.word_id][0] if entry.word_id in best_sim_by_word else -1
                    if score > prev_score:
                        best_sim_by_word[entry.word_id] = (score, sim_frames)
        finally:
            hand_det.close()
            pose_det.close()

        rng = np.random.default_rng(args.seed)
        seqs: list[np.ndarray] = []
        labels: list[str] = []
        groups: list[str] = []
        clips_by_word: dict[str, list[tuple[str, list[np.ndarray]]]] = {}

        for word_id, vname, raw_frames in clean_clips:
            clips_by_word.setdefault(word_id, []).append((vname, raw_frames))
            seqs.append(prepare_wl_model_input(raw_frames)[0])
            labels.append(word_id)
            groups.append(f"{word_id}_{Path(vname).stem}")

        if args.augment_per_class > 0:
            for word_id in classes:
                w_clips = clips_by_word[word_id]
                for aug_i in range(args.augment_per_class):
                    vname, raw_frames = w_clips[aug_i % len(w_clips)]
                    aug_raw = augment_wl_raw_frames(
                        raw_frames,
                        rng=rng,
                        mirror=bool(rng.random() < 0.5),
                        x_scale=float(rng.uniform(0.55, 1.45)),
                        y_scale=float(rng.uniform(0.85, 1.15)),
                        rot_deg=float(rng.uniform(-10.0, 10.0)),
                        crop_start=float(rng.uniform(0.0, 0.30)),
                        crop_end=float(rng.uniform(0.82, 1.0)),
                        noise_std=0.008,
                    )
                    seqs.append(prepare_wl_model_input(aug_raw)[0])
                    labels.append(word_id)
                    groups.append(f"{word_id}_{Path(vname).stem}_aug{aug_i}")

        X_arr = np.stack(seqs, axis=0)
        y_arr = np.array(labels, dtype=str)
        g_arr = np.array(groups, dtype=str)
        save_wl_sequences_npz(custom_npz, X_arr, y_arr, groups=g_arr)
        print(f"Disimpan {len(X_arr)} sequence ({len(clean_clips)} klip bersih) -> {custom_npz}")

        _, w0_tflite, _ = extract_wl_backbone_embeddings(wl_tflite, X_arr[:1])
        patch_tflite = wl_tflite if len(classes) == w0_tflite.shape[0] else None

        w_head, b_head, train_acc = train_wl_custom_head(
            base_tflite_path=wl_tflite,
            classes=classes,
            X_seq=X_arr,
            y_labels=y_arr,
            epochs=args.epochs,
            learning_rate=args.lr,
            l2_reg=args.l2_reg,
            seed=args.seed,
            patch_tflite_path=patch_tflite,
        )
        print(f"Akurasi training head ({len(classes)} kelas): {train_acc:.2%}")

        clean_X = np.stack([prepare_wl_model_input(rf)[0] for _, _, rf in clean_clips], axis=0)
        clean_preds = predict_wl_with_head(wl_tflite, clean_X, w_head, b_head, classes)
        clean_ok = 0
        print("\n=== Evaluasi Klip Bersih ===")
        for (word_id, vname, _), preds in zip(clean_clips, clean_preds):
            top = preds[0]
            ok = top.word == word_id
            if ok:
                clean_ok += 1
            print(f"  [{'OK' if ok else 'MISS'}] {word_id:16s} ({vname}) -> {top.word} ({top.confidence:.1%})")
        print(f"Akurasi klip bersih: {clean_ok}/{len(clean_clips)} ({clean_ok / len(clean_clips):.2%})")

        pub_dir.mkdir(parents=True, exist_ok=True)
        export_wl_browser_weights(wl_tflite, pub_dir, head_override=(w_head, b_head))
        write_wl_labels_json(wl_labels, classes)
        write_wl_labels_json(pub_dir / wl_labels.name, classes)
        if patch_tflite is not None:
            pub_wl = pub_dir / wl_tflite.name
            if pub_wl.resolve() != wl_tflite.resolve():
                shutil.copy2(wl_tflite, pub_wl)
        else:
            print(
                f"[INFO] Jumlah kelas = {len(classes)} (!= {w0_tflite.shape[0]}), "
                f"{wl_tflite} dipertahankan sebagai backbone tanpa menimpa TFLite 8-kelas; "
                f"model browser ({pub_dir / 'wl_model.json'} & {pub_dir / 'wl_weights.bin'}) sudah diperbarui."
            )

        sim_map = {entry.word_id: best_sim_by_word[entry.word_id][1] for entry in entries}
        args.sim_sequences_out.parent.mkdir(parents=True, exist_ok=True)
        args.sim_sequences_out.write_text(json.dumps(sim_map, separators=(",", ":")), encoding="utf-8")
        print(f"Simulasi rangka diperbarui -> {args.sim_sequences_out}")
    else:
        print("[DEMO-ONLY] Melewati pelatihan ulang model AI; hanya memperbarui video demo & katalog UI.")

    keep_mp4s = {f"{entry.word_id}.mp4" for entry in entries}
    bundle_videos_dir = args.bundle_dir / "videos"
    encoded_demos = []
    if not args.skip_video_encode:
        args.public_videos_dir.mkdir(parents=True, exist_ok=True)
        bundle_videos_dir.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            spec = WordDemoSpec(word_id=entry.word_id, title=entry.title)
            demo = encode_word_demo(
                spec,
                entry.representative_video,
                args.public_videos_dir,
                fmt="mp4",
            )
            shutil.copy2(demo.output_path, bundle_videos_dir / demo.object_name)
            encoded_demos.append(demo)

        sync_directory_mp4s(args.public_videos_dir, keep_mp4s)
        sync_directory_mp4s(bundle_videos_dir, keep_mp4s)
        ps_script = render_powershell_upload_script(
            [WordDemoSpec(word_id=e.word_id, title=e.title) for e in entries],
            prune_stale=True,
            default_bucket=args.bucket,
            default_table=args.table,
        )
        args.bundle_dir.mkdir(parents=True, exist_ok=True)
        (args.bundle_dir / "upload_word_demos.ps1").write_text(ps_script, encoding="utf-8")
        print(
            f"Video demo lokal & bundle PowerShell diperbarui ({len(encoded_demos)} video) -> "
            f"{args.public_videos_dir} & {args.bundle_dir}"
        )

    frontend_words: list[dict[str, Any]] = []
    for entry in entries:
        sim_frames = best_sim_by_word.get(entry.word_id, (0, []))[1]
        if entry.hands_override in (1, 2):
            hands = entry.hands_override
        elif sim_frames:
            hands = infer_hands_count(sim_frames)
        else:
            hands = 2
        guide = entry.guide_override or build_default_guide(entry.title, hands)
        frontend_words.append(
            {
                "id": entry.word_id,
                "title": entry.title,
                "hands": hands,
                "guide": guide,
                "localVideoPath": f"/videos/bisindo/{entry.word_id}.mp4",
            }
        )

    write_frontend_words_json(args.frontend_words_json, frontend_words, model_version=model_version)
    print(f"Katalog website diperbarui (modelVersion={model_version}) -> {args.frontend_words_json}")

    if args.upload_supabase:
        if not encoded_demos:
            raise RuntimeError("--upload-supabase tidak bisa digabung dengan --skip-video-encode")
        supabase_url, service_key = _resolve_supabase_credentials(args.env_file)
        if not supabase_url or not service_key:
            print(
                "Error: NEXT_PUBLIC_SUPABASE_URL dan SUPABASE_SERVICE_ROLE_KEY belum diset "
                "di environment atau .env.local.",
                file=sys.stderr,
            )
            return 1

        rows = []
        for demo in encoded_demos:
            public_url = upload_file_to_supabase_storage(
                supabase_url=supabase_url,
                service_role_key=service_key,
                bucket=args.bucket,
                object_name=demo.object_name,
                file_path=demo.output_path,
                mime_type=demo.mime_type,
            )
            rows.append(build_word_demo_row(demo, supabase_url, args.bucket))
            print(f"  [Supabase OK] {demo.object_name} -> {public_url}")

        upsert_word_demo_rows(
            supabase_url=supabase_url,
            service_role_key=service_key,
            rows=rows,
            table=args.table,
        )
        delete_stale_word_demo_rows(
            supabase_url=supabase_url,
            service_role_key=service_key,
            keep_word_ids=classes,
            table=args.table,
        )
        print(f"Tabel public.{args.table} diperbarui ({len(rows)} kata, kata lama dihapus).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
