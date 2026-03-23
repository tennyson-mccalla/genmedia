import json
import os
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from genmedia.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


class TestImageCommand:
    def test_missing_api_key(self, runner):
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(cli, ["image", "a cat"])
        assert result.exit_code == 2
        err = json.loads(result.stderr)
        assert err["error"] == "validation_error"
        assert "GEMINI_API_KEY" in err["message"]

    def test_empty_prompt(self, runner, mock_env):
        result = runner.invoke(cli, ["image", ""])
        assert result.exit_code == 2

    def test_dry_run(self, runner, mock_env):
        result = runner.invoke(cli, ["image", "a cat", "--dry-run"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "dry_run"
        assert parsed["backend"] == "GeminiImageBackend"
        assert parsed["model"] == "gemini-3.1-flash-image-preview"

    def test_dry_run_imagen(self, runner, mock_env):
        result = runner.invoke(cli, ["image", "a cat", "--model", "imagen-4.0-generate-001", "--dry-run"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["backend"] == "ImagenBackend"

    def test_dry_run_with_validation_errors(self, runner, mock_env):
        result = runner.invoke(cli, ["image", "a cat", "--aspect", "99:1", "--dry-run"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed["validation_errors"]) > 0

    def test_list_models(self, runner):
        # Does not require API key
        result = runner.invoke(cli, ["image", "--list-models"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "models" in parsed
        assert any(m["default"] for m in parsed["models"])

    def test_invalid_aspect_ratio(self, runner, mock_env):
        result = runner.invoke(cli, ["image", "a cat", "--aspect", "7:3"])
        assert result.exit_code == 2
        err = json.loads(result.stderr)
        assert err["error"] == "validation_error"

    @patch("genmedia.cli.image.genai")
    def test_successful_generation(self, mock_genai, runner, mock_env, tmp_path):
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"fake-png"
        mock_part.inline_data.mime_type = "image/png"
        mock_part.text = None

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.candidates[0].content.parts = [mock_part]

        mock_genai.Client.return_value.models.generate_content.return_value = mock_response

        result = runner.invoke(cli, [
            "image", "a cat",
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"
        assert len(parsed["files"]) == 1
        assert os.path.isfile(parsed["files"][0]["path"])


class TestEditCommand:
    def test_dry_run_edit(self, runner, mock_env, tmp_path):
        # Create a fake input image
        img_path = tmp_path / "input.png"
        img_path.write_bytes(b"fake-png-data")

        result = runner.invoke(cli, ["edit", str(img_path), "remove background", "--dry-run"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "dry_run"
        assert parsed["backend"] == "GeminiImageBackend"
        assert parsed["config"]["response_modalities"] == ["TEXT", "IMAGE"]

    def test_edit_missing_input_file(self, runner, mock_env):
        result = runner.invoke(cli, ["edit", "/nonexistent.png", "remove bg"])
        assert result.exit_code == 2

    def test_edit_missing_prompt(self, runner, mock_env, tmp_path):
        img_path = tmp_path / "input.png"
        img_path.write_bytes(b"fake-png")
        result = runner.invoke(cli, ["edit", str(img_path)])
        assert result.exit_code != 0

    @patch("genmedia.cli.edit.genai")
    def test_edit_success(self, mock_genai, runner, mock_env, tmp_path):
        img_path = tmp_path / "input.png"
        img_path.write_bytes(b"fake-input")

        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"edited-png"
        mock_part.inline_data.mime_type = "image/png"
        mock_part.text = None

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.candidates[0].content.parts = [mock_part]

        mock_genai.Client.return_value.models.generate_content.return_value = mock_response

        result = runner.invoke(cli, [
            "edit", str(img_path), "remove background",
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"


class TestVideoCommand:
    def test_dry_run_video(self, runner, mock_env):
        result = runner.invoke(cli, ["video", "a sunset", "--dry-run"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "dry_run"
        assert parsed["backend"] == "VeoBackend"
        assert parsed["model"] == "veo-3.0-generate-001"

    def test_dry_run_with_duration(self, runner, mock_env):
        result = runner.invoke(cli, ["video", "a sunset", "--duration", "8", "--dry-run"])
        parsed = json.loads(result.output)
        assert parsed["config"]["duration_seconds"] == 8

    def test_invalid_duration(self, runner, mock_env):
        result = runner.invoke(cli, ["video", "a sunset", "--duration", "5"])
        assert result.exit_code == 2
        err = json.loads(result.stderr)
        assert err["error"] == "validation_error"

    def test_list_video_models(self, runner):
        result = runner.invoke(cli, ["video", "--list-models"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert any(m["id"].startswith("veo") for m in parsed["models"])

    @patch("genmedia.cli.video.genai")
    def test_video_success(self, mock_genai, runner, mock_env, tmp_path):
        mock_video = MagicMock()
        mock_video.video = MagicMock()

        mock_operation = MagicMock()
        mock_operation.done = True
        mock_operation.result.generated_videos = [mock_video]

        mock_genai.Client.return_value.models.generate_videos.return_value = mock_operation
        mock_genai.Client.return_value.files.download.return_value = b"fake-video"

        result = runner.invoke(cli, [
            "video", "a sunset",
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"
        assert parsed["files"][0]["mime_type"] == "video/mp4"
