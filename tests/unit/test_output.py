import json
import os
import tempfile

from genmedia.output import (
    format_success, format_error, format_dry_run,
    format_list_models, auto_name, write_media_files,
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
