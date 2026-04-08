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


class TestAdditionalCoverage:
    def test_explicit_output_path(self, runner, mock_env, tmp_path):
        """Test --output flag writes to exact path."""
        with patch("genmedia.cli.image.genai") as mock_genai:
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

            out_path = str(tmp_path / "exact.png")
            result = runner.invoke(cli, ["image", "a cat", "--output", out_path])
            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert parsed["files"][0]["path"] == out_path
            assert os.path.isfile(out_path)

    def test_count_multiple_files(self, runner, mock_env, tmp_path):
        """Test --count 3 produces 3 files."""
        with patch("genmedia.cli.image.genai") as mock_genai:
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
                "image", "a cat", "--count", "3", "--output-dir", str(tmp_path),
            ])
            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert len(parsed["files"]) == 3
            for f in parsed["files"]:
                assert os.path.isfile(f["path"])

    def test_format_jpg(self, runner, mock_env, tmp_path):
        """Test --format jpg produces .jpg extension."""
        with patch("genmedia.cli.image.genai") as mock_genai:
            mock_part = MagicMock()
            mock_part.inline_data = MagicMock()
            mock_part.inline_data.data = b"fake-jpg"
            mock_part.inline_data.mime_type = "image/jpeg"
            mock_part.text = None

            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_response.candidates[0].finish_reason = "STOP"
            mock_response.candidates[0].content.parts = [mock_part]

            mock_genai.Client.return_value.models.generate_content.return_value = mock_response

            result = runner.invoke(cli, [
                "image", "a cat", "--format", "jpg", "--output-dir", str(tmp_path),
            ])
            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert parsed["files"][0]["path"].endswith(".jpg")


class TestVideoImageToVideo:
    def test_dry_run_with_image(self, runner, mock_env, tmp_path):
        img = tmp_path / "frame.png"
        img.write_bytes(b"fake-frame")
        result = runner.invoke(cli, ["video", "animate this", "--image", str(img), "--dry-run"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["config"]["image"] == str(img)

    def test_dry_run_with_image_and_last_frame(self, runner, mock_env, tmp_path):
        first = tmp_path / "first.png"
        last = tmp_path / "last.png"
        first.write_bytes(b"frame-a")
        last.write_bytes(b"frame-b")
        result = runner.invoke(cli, [
            "video", "morph between frames",
            "--image", str(first), "--last-frame", str(last), "--dry-run",
        ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["config"]["image"] == str(first)
        assert parsed["config"]["last_frame"] == str(last)

    def test_last_frame_without_image_fails(self, runner, mock_env, tmp_path):
        last = tmp_path / "last.png"
        last.write_bytes(b"frame-b")
        result = runner.invoke(cli, ["video", "test", "--last-frame", str(last)])
        assert result.exit_code == 2

    def test_image_only_no_prompt(self, runner, mock_env, tmp_path):
        img = tmp_path / "frame.png"
        img.write_bytes(b"fake-frame")
        result = runner.invoke(cli, ["video", "--image", str(img), "--dry-run"])
        assert result.exit_code == 0

    @patch("genmedia.cli.video.genai")
    def test_image_to_video_success(self, mock_genai, runner, mock_env, tmp_path):
        img = tmp_path / "frame.png"
        img.write_bytes(b"fake-frame")

        mock_video = MagicMock()
        mock_video.video = MagicMock()

        mock_operation = MagicMock()
        mock_operation.done = True
        mock_operation.result.generated_videos = [mock_video]

        mock_genai.Client.return_value.models.generate_videos.return_value = mock_operation
        mock_genai.Client.return_value.files.download.return_value = b"fake-video"

        result = runner.invoke(cli, [
            "video", "animate this", "--image", str(img),
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"
        assert parsed["request"]["image"] == str(img)


class TestVideoV020Features:
    def test_dry_run_with_resolution(self, runner, mock_env):
        result = runner.invoke(cli, ["video", "test", "--resolution", "1080p", "--dry-run"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["config"]["resolution"] == "1080p"

    def test_dry_run_with_enhance_prompt(self, runner, mock_env):
        result = runner.invoke(cli, ["video", "test", "--enhance-prompt", "--dry-run"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["config"]["enhance_prompt"] is True

    def test_resolution_case_insensitive(self, runner, mock_env):
        result = runner.invoke(cli, ["video", "test", "--resolution", "1080p", "--duration", "8", "--dry-run"])
        assert result.exit_code == 0

    def test_verbose_flag_removed(self, runner, mock_env):
        result = runner.invoke(cli, ["image", "test", "-v", "--dry-run"])
        assert result.exit_code != 0  # -v is no longer recognized

    def test_dry_run_with_style_ref(self, runner, mock_env, tmp_path):
        ref = tmp_path / "style.png"
        ref.write_bytes(b"style-image")
        result = runner.invoke(cli, ["video", "test prompt", "--style-ref", str(ref), "--dry-run"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["config"]["style_ref"] == str(ref)

    def test_dry_run_with_asset_refs(self, runner, mock_env, tmp_path):
        a1 = tmp_path / "asset1.png"
        a2 = tmp_path / "asset2.png"
        a1.write_bytes(b"asset-1")
        a2.write_bytes(b"asset-2")
        result = runner.invoke(cli, [
            "video", "test prompt",
            "--asset-ref", str(a1), "--asset-ref", str(a2),
            "--dry-run",
        ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed["config"]["asset_refs"]) == 2

    def test_style_ref_and_asset_ref_conflict(self, runner, mock_env, tmp_path):
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"ref")
        result = runner.invoke(cli, [
            "video", "test", "--style-ref", str(ref), "--asset-ref", str(ref),
        ])
        assert result.exit_code == 2

    def test_style_ref_with_image_conflict(self, runner, mock_env, tmp_path):
        ref = tmp_path / "ref.png"
        img = tmp_path / "img.png"
        ref.write_bytes(b"ref")
        img.write_bytes(b"img")
        result = runner.invoke(cli, [
            "video", "test", "--style-ref", str(ref), "--image", str(img),
        ])
        assert result.exit_code == 2

    def test_style_ref_requires_prompt(self, runner, mock_env, tmp_path):
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"ref")
        result = runner.invoke(cli, ["video", "--style-ref", str(ref)])
        assert result.exit_code == 2


class TestStdinStdoutPiping:
    @patch("genmedia.cli.image.genai")
    def test_stdout_output(self, mock_genai, runner, mock_env):
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"\x89PNG\r\n\x1a\nfake"
        mock_part.inline_data.mime_type = "image/png"
        mock_part.text = None

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.candidates[0].content.parts = [mock_part]

        mock_genai.Client.return_value.models.generate_content.return_value = mock_response

        result = runner.invoke(cli, ["image", "a cat", "--output", "-"])
        assert result.exit_code == 0

    def test_stdin_prompt_for_image(self, runner, mock_env):
        with patch("genmedia.cli.image.genai"):
            result = runner.invoke(cli, ["image", "--dry-run"], input="a piped prompt\n")
            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert parsed["status"] == "dry_run"

    def test_stdin_prompt_for_video(self, runner, mock_env):
        result = runner.invoke(cli, ["video", "--dry-run"], input="a piped video prompt\n")
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "dry_run"


class TestVideoTimeout:
    @patch("genmedia.cli.video.genai")
    def test_poll_timeout_produces_clean_error(self, mock_genai, runner, mock_env):
        pending_op = MagicMock()
        pending_op.done = False

        mock_genai.Client.return_value.models.generate_videos.return_value = pending_op
        mock_genai.Client.return_value.operations.get.return_value = pending_op

        with patch.dict(os.environ, {"GENMEDIA_POLL_TIMEOUT": "0.1"}):
            with patch("genmedia.backends.veo.time.sleep"):
                result = runner.invoke(cli, ["video", "a sunset"])

        assert result.exit_code == 1
        err = json.loads(result.stderr)
        assert err["error"] == "timeout"
        assert "timed out" in err["message"]
