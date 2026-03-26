import json
import os
import tempfile

from click.testing import CliRunner

from genmedia.cli.main import cli


def test_imagen_happy_path():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(cli, [
            "image", "a simple blue square",
            "--model", "imagen-4.0-generate-001",
            "--output-dir", tmpdir,
        ])
        assert result.exit_code == 0, f"Failed: {result.output}"
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"
        assert len(parsed["files"]) == 1
        assert parsed["files"][0]["size_bytes"] > 0
        assert os.path.isfile(parsed["files"][0]["path"])
