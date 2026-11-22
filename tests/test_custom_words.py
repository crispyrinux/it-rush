"""Unit tests for custom BISINDO word discovery, arbitrary-class WL training, and export."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bisindo.custom_words import (
    discover_custom_word_videos,
    format_word_title,
    infer_hands_count,
    normalize_word_id,
    sync_directory_mp4s,
    write_frontend_words_json,
    write_wl_labels_json,
)
from bisindo.word_demos import (
    WordDemoSpec,
    delete_stale_word_demo_rows,
    render_powershell_upload_script,
)
from bisindo.word_level import (
    WL_SEQUENCE_LEN,
    WL_TEMPORAL_FEATURE_DIM,
    WordLevelPredictor,
    export_wl_browser_weights,
    predict_wl_with_head,
    train_wl_custom_head,
)


def test_normalize_word_id_and_format_title():
    assert normalize_word_id("pantai.mp4", is_filename=True) == "pantai"
    assert normalize_word_id("Terima Kasih.mp4", is_filename=True) == "terima_kasih"
    assert normalize_word_id("Pantai-Laut__take2.mp4", is_filename=True) == "pantai_laut"
    assert normalize_word_id("  Hutan   Bakau! ") == "hutan_bakau"
    assert format_word_title("terima_kasih") == "Terima Kasih"

    with pytest.raises(ValueError, match="word_id"):
        normalize_word_id("___")
    with pytest.raises(ValueError, match="word_id"):
        normalize_word_id("...mp4", is_filename=True)


def test_discover_custom_word_videos_mixed_layout_and_meta(tmp_path: Path):
    # Flat files, multi-take __ files, subfolder clips, and ignored non-video files
    (tmp_path / "README.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / "hutan.mp4").write_bytes(b"12345")
    (tmp_path / "hutan__take2.mp4").write_bytes(b"123456789")
    (tmp_path / "terima_kasih.mp4").write_bytes(b"1234")

    laut_dir = tmp_path / "laut"
    laut_dir.mkdir()
    (laut_dir / "clip_small.mp4").write_bytes(b"12")
    (laut_dir / "clip_large.mp4").write_bytes(b"12345678")

    (tmp_path / "words_meta.json").write_text(
        json.dumps(
            {
                "laut": {"title": "Laut Biru", "hands": 1, "guide": "Gerakkan 1 tangan.", "order": 1},
                "hutan": {"order": 2},
            }
        ),
        encoding="utf-8",
    )

    entries = discover_custom_word_videos(tmp_path)
    assert [e.word_id for e in entries] == ["laut", "hutan", "terima_kasih"]
    assert entries[0].title == "Laut Biru"
    assert entries[0].hands_override == 1
    assert entries[0].guide_override == "Gerakkan 1 tangan."
    assert entries[0].representative_video == laut_dir / "clip_large.mp4"
    assert entries[1].representative_video == tmp_path / "hutan.mp4"
    assert len(entries[1].video_paths) == 2
    assert entries[2].title == "Terima Kasih"


def test_discover_custom_word_videos_requires_at_least_two_words(tmp_path: Path):
    (tmp_path / "tunggal.mp4").write_bytes(b"123")
    with pytest.raises(ValueError, match="Minimal 2 kata"):
        discover_custom_word_videos(tmp_path)


def test_infer_hands_count():
    one_hand = [{"hands": [[[0.1, 0.2]]]} for _ in range(10)]
    two_hands = [{"hands": [[[0.1, 0.2]], [[0.3, 0.4]]]} for _ in range(6)] + [
        {"hands": [[[0.1, 0.2]]]} for _ in range(4)
    ]
    assert infer_hands_count(one_hand) == 1
    assert infer_hands_count(two_hands) == 2
    assert infer_hands_count([{"hands": []}, {"hands": []}]) == 2


def test_train_custom_head_and_export_browser_weights_non8_and_8(tmp_path: Path):
    base_tflite = Path("artifacts/bisindo_wl_model.tflite")
    if not base_tflite.exists():
        pytest.skip("bisindo_wl_model.tflite not available")

    copied_backbone = tmp_path / "backbone.tflite"
    original_bytes = base_tflite.read_bytes()
    copied_backbone.write_bytes(original_bytes)

    rng = np.random.default_rng(123)
    classes_5 = ["ombak", "hutan", "karang", "penyu", "mangrove"]
    X_5 = rng.normal(0.0, 0.35, size=(10, WL_SEQUENCE_LEN, WL_TEMPORAL_FEATURE_DIM)).astype(
        np.float32
    )
    y_5 = np.array(classes_5 * 2)

    w_head, b_head, acc = train_wl_custom_head(
        copied_backbone,
        classes_5,
        X_5,
        y_5,
        epochs=15,
        patch_tflite_path=copied_backbone,
    )
    assert w_head.shape == (5, 128)
    assert b_head.shape == (5,)
    assert 0.0 <= acc <= 1.0
    # Must NOT mutate or corrupt 8-class TFLite when N != 8
    assert copied_backbone.read_bytes() == original_bytes

    preds_5 = predict_wl_with_head(copied_backbone, X_5[:2], w_head, b_head, classes_5)
    assert len(preds_5) == 2
    assert len(preds_5[0]) == 5
    assert np.isclose(sum(p.confidence for p in preds_5[0]), 1.0, atol=1e-4)

    out_dir = tmp_path / "browser"
    json_path, bin_path = export_wl_browser_weights(
        copied_backbone,
        out_dir,
        head_override=(w_head, b_head),
    )
    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    raw_bin = bin_path.read_bytes()

    t40 = manifest["tensors"]["40"]
    t59 = manifest["tensors"]["59"]
    assert t40["shape"] == [5, 128]
    assert t59["shape"] == [5]

    exported_w = np.frombuffer(
        raw_bin[t40["byteOffset"] : t40["byteOffset"] + t40["byteLength"]],
        dtype="<f4",
    ).reshape(5, 128)
    exported_b = np.frombuffer(
        raw_bin[t59["byteOffset"] : t59["byteOffset"] + t59["byteLength"]],
        dtype="<f4",
    )
    assert np.allclose(exported_w, w_head, atol=0.0)
    assert np.allclose(exported_b, b_head, atol=0.0)

    # Verify 8-class path patches TFLite cleanly as well
    classes_8 = ["satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan"]
    X_8 = rng.normal(0.0, 0.35, size=(16, WL_SEQUENCE_LEN, WL_TEMPORAL_FEATURE_DIM)).astype(
        np.float32
    )
    y_8 = np.array(classes_8 * 2)
    patched_8 = tmp_path / "patched8.tflite"
    train_wl_custom_head(
        copied_backbone,
        classes_8,
        X_8,
        y_8,
        epochs=10,
        patch_tflite_path=patched_8,
    )
    labels_8 = write_wl_labels_json(tmp_path / "labels8.json", classes_8)
    predictor = WordLevelPredictor(patched_8, labels_8)
    assert predictor.index_to_label[7] == "delapan"


def test_sync_directory_mp4s_render_ps1_and_delete_stale(tmp_path: Path):
    (tmp_path / "old_word.mp4").write_bytes(b"old")
    (tmp_path / "keep_word.mp4").write_bytes(b"keep")
    (tmp_path / "notes.txt").write_text("stay", encoding="utf-8")

    removed = sync_directory_mp4s(tmp_path, {"keep_word.mp4"})
    assert [p.name for p in removed] == ["old_word.mp4"]
    assert (tmp_path / "keep_word.mp4").exists()
    assert (tmp_path / "notes.txt").exists()

    ps1 = render_powershell_upload_script(
        [WordDemoSpec(word_id="ombak", title="Ombak"), WordDemoSpec(word_id="penyu", title="Penyu")]
    )
    assert 'word_id = "ombak"' in ps1
    assert "word_id=not.in.(ombak,penyu)" in ps1

    with pytest.raises(ValueError, match="Invalid word_id"):
        render_powershell_upload_script([WordDemoSpec(word_id="bad;id", title="Bad")])

    words_json = write_frontend_words_json(
        tmp_path / "bisindoWords.json",
        [
            {
                "id": "ombak",
                "title": "Ombak",
                "hands": 2,
                "guide": "Panduan",
                "localVideoPath": "/videos/bisindo/ombak.mp4",
            }
        ],
        model_version="custom-test",
    )
    parsed = json.loads(words_json.read_text(encoding="utf-8"))
    assert parsed["modelVersion"] == "custom-test"
    assert parsed["words"][0]["id"] == "ombak"

    mock_resp = MagicMock()
    mock_resp.status = 204
    mock_resp.read.return_value = b""
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    with patch("bisindo.word_demos.urllib.request.urlopen", return_value=mock_resp) as mock_open:
        delete_stale_word_demo_rows(
            supabase_url="https://demo.supabase.co",
            service_role_key="service-role-secret",
            keep_word_ids=["ombak", "penyu"],
        )
        req = mock_open.call_args[0][0]
        assert req.get_method() == "DELETE"
        assert (
            req.full_url
            == "https://demo.supabase.co/rest/v1/bisindo_word_demos?word_id=not.in.(ombak,penyu)"
        )
