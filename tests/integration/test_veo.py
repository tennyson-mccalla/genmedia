import json
import os
import tempfile

from click.testing import CliRunner

from genmedia.cli.main import cli


def test_veo_happy_path():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(cli, [
            "video", "a slow pan across a calm lake at dawn",
            "--duration", "4",
            "--output-dir", tmpdir,
        ])
        assert result.exit_code == 0, f"Failed: {result.output}"
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"
        assert len(parsed["files"]) == 1
        assert parsed["files"][0]["mime_type"] == "video/mp4"
        assert parsed["files"][0]["size_bytes"] > 0
        assert os.path.isfile(parsed["files"][0]["path"])
