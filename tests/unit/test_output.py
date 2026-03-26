import json
import os
import tempfile

from genmedia.output import (
    format_success, format_error, format_dry_run,
    format_list_models, auto_name, write_media_files,
    detect_mime_type,
)


def test_format_success_json():
    result = format_success(
        files=[{"path": "/tmp/genmedia/genmedia_001.png", "mime_type": "image/png", "size_bytes": 1000}],
        model="gemini-3.1-flash-image-preview",
        elapsed_seconds=2.5,
        request={"prompt": "cat", "aspect_ratio": "16:9"},
    )
    parsed = json.loads(result)
    assert parsed["status"] == "success"
    assert len(parsed["files"]) == 1
    assert parsed["model"] == "gemini-3.1-flash-image-preview"
    assert parsed["elapsed_seconds"] == 2.5


def test_format_error_json():
    result = format_error(
        error="rate_limited", message="429 after 5 retries",
        retries_attempted=5, elapsed_seconds=30.0,
    )
    parsed = json.loads(result)
    assert parsed["status"] == "error"
    assert parsed["error"] == "rate_limited"
    assert parsed["retries_attempted"] == 5


def test_format_error_with_partial_files():
    result = format_error(
        error="file_error", message="Disk full",
        retries_attempted=0, elapsed_seconds=5.0,
        files=[{"path": "/tmp/genmedia/genmedia_001.png", "mime_type": "image/png", "size_bytes": 1000}],
    )
    parsed = json.loads(result)
    assert parsed["status"] == "error"
    assert len(parsed["files"]) == 1


def test_format_dry_run_json():
    result = format_dry_run(
        backend="GeminiImageBackend", sdk_method="client.models.generate_content",
        model="gemini-3.1-flash-image-preview",
        config={"response_modalities": ["IMAGE"]}, validation_errors=[],
    )
    parsed = json.loads(result)
    assert parsed["status"] == "dry_run"
    assert parsed["backend"] == "GeminiImageBackend"
    assert parsed["validation_errors"] == []


def test_format_dry_run_with_errors():
    result = format_dry_run(
        backend="GeminiImageBackend", sdk_method="client.models.generate_content",
        model="gemini-3.1-flash-image-preview",
        config={}, validation_errors=["bad aspect ratio"],
    )
    parsed = json.loads(result)
    assert parsed["validation_errors"] == ["bad aspect ratio"]


def test_format_list_models_json():
    models = [{"id": "model-a", "default": True, "notes": "best"}, {"id": "model-b", "default": False, "notes": "fast"}]
    result = format_list_models(models)
    parsed = json.loads(result)
    assert len(parsed["models"]) == 2


def test_auto_name_no_collision():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = auto_name(output_dir=tmpdir, extension=".png")
        assert path.endswith(".png")
        assert "genmedia_001" in path


def test_auto_name_collision_avoidance():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "genmedia_001.png"), "w").close()
        path = auto_name(output_dir=tmpdir, extension=".png")
        assert "genmedia_002" in path


def test_auto_name_creates_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "nested", "dir")
        path = auto_name(output_dir=subdir, extension=".png")
        assert os.path.isdir(subdir)


def test_write_media_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        from genmedia.backends.base import MediaResult
        results = [MediaResult(data=b"fake-png-data", mime_type="image/png", metadata={})]
        written = write_media_files(results=results, output=None, output_dir=tmpdir, output_format="png")
        assert len(written) == 1
        assert os.path.isfile(written[0]["path"])
        assert written[0]["size_bytes"] == len(b"fake-png-data")


# --- MIME type detection from bytes ---

# Real magic bytes for each format
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 100
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
WEBP_BYTES = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 100
MP4_BYTES = b"\x00\x00\x00\x1cftyp" + b"\x00" * 100


def test_detect_jpeg():
    assert detect_mime_type(JPEG_BYTES) == "image/jpeg"


def test_detect_png():
    assert detect_mime_type(PNG_BYTES) == "image/png"


def test_detect_webp():
    assert detect_mime_type(WEBP_BYTES) == "image/webp"


def test_detect_mp4():
    assert detect_mime_type(MP4_BYTES) == "video/mp4"


def test_detect_unknown_returns_none():
    assert detect_mime_type(b"random garbage data") is None


def test_detect_empty_returns_none():
    assert detect_mime_type(b"") is None


def test_write_media_files_corrects_extension_for_jpeg():
    """When API returns JPEG bytes but user asked for png, file gets .jpg extension."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from genmedia.backends.base import MediaResult
        results = [MediaResult(data=JPEG_BYTES, mime_type="image/png", metadata={})]
        written = write_media_files(results=results, output=None, output_dir=tmpdir, output_format="png")
        assert written[0]["path"].endswith(".jpg")
        assert written[0]["mime_type"] == "image/jpeg"


def test_write_media_files_corrects_mime_type():
    """The reported mime_type should match actual bytes, not what the API claimed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from genmedia.backends.base import MediaResult
        results = [MediaResult(data=PNG_BYTES, mime_type="image/jpeg", metadata={})]
        written = write_media_files(results=results, output=None, output_dir=tmpdir, output_format="jpg")
        assert written[0]["path"].endswith(".png")
        assert written[0]["mime_type"] == "image/png"


def test_write_media_files_explicit_output_keeps_path():
    """When user provides --output, the path is honored but mime_type is still corrected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from genmedia.backends.base import MediaResult
        explicit_path = os.path.join(tmpdir, "my_image.png")
        results = [MediaResult(data=JPEG_BYTES, mime_type="image/png", metadata={})]
        written = write_media_files(results=results, output=explicit_path, output_dir=None, output_format="png")
        # Path is what user asked for — we don't rename explicit paths
        assert written[0]["path"].endswith("my_image.png")
        # But mime_type is corrected
        assert written[0]["mime_type"] == "image/jpeg"


def test_write_media_files_stdout():
    """--output - writes binary to stdout."""
    import io
    import sys
    from unittest.mock import patch
    from genmedia.backends.base import MediaResult

    results = [MediaResult(data=PNG_BYTES, mime_type="image/png", metadata={})]
    fake_stdout = io.BytesIO()
    with patch.object(sys, "stdout", wraps=sys.stdout) as mock_stdout:
        mock_stdout.buffer = fake_stdout
        written = write_media_files(results=results, output="-", output_dir=None, output_format="png")
    assert written[0]["path"] == "-"
    assert written[0]["mime_type"] == "image/png"
    assert fake_stdout.getvalue() == PNG_BYTES


def test_write_media_files_stdout_rejects_multiple():
    """--output - with multiple results should raise ValueError."""
    import pytest
    from genmedia.backends.base import MediaResult

    results = [
        MediaResult(data=PNG_BYTES, mime_type="image/png", metadata={}),
        MediaResult(data=PNG_BYTES, mime_type="image/png", metadata={}),
    ]
    with pytest.raises(ValueError, match="single file"):
        write_media_files(results=results, output="-", output_dir=None, output_format="png")
