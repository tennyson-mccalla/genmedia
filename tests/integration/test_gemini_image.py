import json
import os
import tempfile

from click.testing import CliRunner

from genmedia.cli.main import cli


def test_gemini_image_happy_path():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(cli, [
            "image", "a simple red circle on white background",
            "--output-dir", tmpdir,
            "--size", "512",
        ])
        assert result.exit_code == 0, f"Failed: {result.output}"
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"
        assert len(parsed["files"]) == 1
        assert os.path.isfile(parsed["files"][0]["path"])
        assert parsed["files"][0]["size_bytes"] > 0
