from genmedia.output import format_pretty_success, format_pretty_error


def test_pretty_success_no_json():
    result = format_pretty_success(
        files=[{"path": "/tmp/genmedia/genmedia_001.png", "mime_type": "image/png", "size_bytes": 1000}],
        model="gemini-3.1-flash-image-preview",
        elapsed_seconds=2.5,
    )
    assert "Saved to" in result
    assert "/tmp/genmedia/genmedia_001.png" in result
    assert "{" not in result  # no JSON


def test_pretty_error_no_json():
    result = format_pretty_error(
        error="rate_limited",
        message="429 after retries",
    )
    assert "Error" in result or "error" in result
    assert "429" in result
    assert "{" not in result  # no JSON


def test_pretty_success_multiple_files():
    files = [
        {"path": "/tmp/genmedia/genmedia_001.png", "mime_type": "image/png", "size_bytes": 1000},
        {"path": "/tmp/genmedia/genmedia_002.png", "mime_type": "image/png", "size_bytes": 1200},
    ]
    result = format_pretty_success(files=files, model="test", elapsed_seconds=3.0)
    assert "genmedia_001" in result
    assert "genmedia_002" in result
