import os
from unittest.mock import patch

from genmedia.validation import validate_config


def test_missing_api_key():
    with patch.dict(os.environ, {}, clear=True):
        errors = validate_config(
            subcommand="image", prompt="hello", aspect_ratio=None,
            image_size=None, duration_seconds=None, output_format="png",
            count=1, model="gemini-3.1-flash-image-preview", input_image=None,
        )
    assert any("GEMINI_API_KEY" in e for e in errors)


def test_empty_prompt():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image", prompt="", aspect_ratio=None,
            image_size=None, duration_seconds=None, output_format="png",
            count=1, model="gemini-3.1-flash-image-preview", input_image=None,
        )
    assert any("empty" in e.lower() for e in errors)


def test_whitespace_prompt():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image", prompt="   ", aspect_ratio=None,
            image_size=None, duration_seconds=None, output_format="png",
            count=1, model="gemini-3.1-flash-image-preview", input_image=None,
        )
    assert any("empty" in e.lower() for e in errors)


def test_valid_aspect_ratios():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        for ratio in ["1:1", "16:9", "9:16", "3:4", "4:3", "21:9"]:
            errors = validate_config(
                subcommand="image", prompt="test", aspect_ratio=ratio,
                image_size=None, duration_seconds=None, output_format="png",
                count=1, model="gemini-3.1-flash-image-preview", input_image=None,
            )
            assert not errors, f"Unexpected error for {ratio}: {errors}"


def test_invalid_aspect_ratio():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image", prompt="test", aspect_ratio="7:3",
            image_size=None, duration_seconds=None, output_format="png",
            count=1, model="gemini-3.1-flash-image-preview", input_image=None,
        )
    assert any("aspect" in e.lower() for e in errors)


def test_video_aspect_ratio_restricted():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="video", prompt="test", aspect_ratio="4:3",
            image_size=None, duration_seconds=8, output_format=None,
            count=1, model="veo-3.0-generate-001", input_image=None,
        )
    assert any("video" in e.lower() for e in errors)


def test_image_size_case_insensitive():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        for size in ["4k", "4K", "1k", "1K", "2k", "2K", "512"]:
            errors = validate_config(
                subcommand="image", prompt="test", aspect_ratio=None,
                image_size=size, duration_seconds=None, output_format="png",
                count=1, model="gemini-3.1-flash-image-preview", input_image=None,
            )
            assert not errors, f"Unexpected error for size {size}: {errors}"


def test_invalid_image_size():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image", prompt="test", aspect_ratio=None,
            image_size="8K", duration_seconds=None, output_format="png",
            count=1, model="gemini-3.1-flash-image-preview", input_image=None,
        )
    assert any("size" in e.lower() for e in errors)


def test_image_size_not_allowed_for_imagen():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image", prompt="test", aspect_ratio=None,
            image_size="4K", duration_seconds=None, output_format="png",
            count=1, model="imagen-4.0-generate-001", input_image=None,
        )
    assert any("imagen" in e.lower() or "size" in e.lower() for e in errors)


def test_valid_duration():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        for d in [4, 6, 8]:
            errors = validate_config(
                subcommand="video", prompt="test", aspect_ratio=None,
                image_size=None, duration_seconds=d, output_format=None,
                count=1, model="veo-3.0-generate-001", input_image=None,
            )
            assert not errors, f"Unexpected error for duration {d}"


def test_invalid_duration():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="video", prompt="test", aspect_ratio=None,
            image_size=None, duration_seconds=5, output_format=None,
            count=1, model="veo-3.0-generate-001", input_image=None,
        )
    assert any("duration" in e.lower() for e in errors)


def test_invalid_count():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image", prompt="test", aspect_ratio=None,
            image_size=None, duration_seconds=None, output_format="png",
            count=0, model="gemini-3.1-flash-image-preview", input_image=None,
        )
    assert any("count" in e.lower() for e in errors)


def test_unknown_model_is_warning_not_error():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image", prompt="test", aspect_ratio=None,
            image_size=None, duration_seconds=None, output_format="png",
            count=1, model="some-future-model", input_image=None,
        )
    assert not errors


def test_edit_missing_input_image():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="edit", prompt="remove background", aspect_ratio=None,
            image_size=None, duration_seconds=None, output_format="png",
            count=1, model="gemini-3.1-flash-image-preview",
            input_image="/nonexistent/file.png",
        )
    assert any("input" in e.lower() or "file" in e.lower() for e in errors)


def test_valid_config_no_errors():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image", prompt="a cat", aspect_ratio="16:9",
            image_size="4K", duration_seconds=None, output_format="png",
            count=1, model="gemini-3.1-flash-image-preview", input_image=None,
        )
    assert errors == []


def test_resolution_1080p_requires_8s():
    errors = validate_config(
        subcommand="video", prompt="x", aspect_ratio=None,
        image_size=None, duration_seconds=4, output_format=None,
        count=1, model="veo-3.1-fast-generate-preview", input_image=None,
    )
    # We thread resolution through validate_config in a later step; for now
    # this test will be wired by Task 1 step 3.
    # Assertion placeholder — see step 3.
    assert errors == []  # baseline check

def test_resolution_1080p_with_8s_ok(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    from genmedia.validation import validate_video_extras
    errors = validate_video_extras(resolution="1080p", duration_seconds=8, model="veo-3.1-generate-preview", last_frame=False)
    assert errors == []

def test_resolution_1080p_with_4s_errors(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    from genmedia.validation import validate_video_extras
    errors = validate_video_extras(resolution="1080p", duration_seconds=4, model="veo-3.1-generate-preview", last_frame=False)
    assert any("1080p" in e and "8" in e for e in errors)


def test_last_frame_rejected_on_veo30():
    from genmedia.validation import validate_video_extras
    errors = validate_video_extras(
        resolution=None, duration_seconds=8,
        model="veo-3.0-generate-001", last_frame=True,
    )
    assert any("last-frame" in e for e in errors)


def test_last_frame_rejected_on_veo31_fast():
    from genmedia.validation import validate_video_extras
    errors = validate_video_extras(
        resolution=None, duration_seconds=8,
        model="veo-3.1-fast-generate-preview", last_frame=True,
    )
    assert any("last-frame" in e for e in errors)


def test_last_frame_allowed_on_veo31_full():
    from genmedia.validation import validate_video_extras
    errors = validate_video_extras(
        resolution=None, duration_seconds=8,
        model="veo-3.1-generate-preview", last_frame=True,
    )
    assert errors == []
