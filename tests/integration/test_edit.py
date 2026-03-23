import json
import os
import tempfile

from click.testing import CliRunner

from genmedia.cli.main import cli


def test_edit_happy_path(tmp_path):
    """Create a simple image first, then edit it."""
    runner = CliRunner(mix_stderr=False)

    # First generate an image to use as input
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(cli, [
            "image", "a solid red square on white background",
            "--output-dir", tmpdir,
            "--size", "512",
        ])
        assert result.exit_code == 0, f"Image gen failed: {result.output}"
        gen_result = json.loads(result.output)
        input_path = gen_result["files"][0]["path"]

        # Now edit it
        result = runner.invoke(cli, [
            "edit", input_path, "make the square blue instead of red",
            "--output-dir", tmpdir,
        ])
        assert result.exit_code == 0, f"Edit failed: {result.output}"
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"
        assert len(parsed["files"]) == 1
        assert os.path.isfile(parsed["files"][0]["path"])
