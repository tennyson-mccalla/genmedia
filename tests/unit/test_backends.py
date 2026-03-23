from unittest.mock import MagicMock, patch
import pytest

from genmedia.backends.base import MediaConfig, MediaResult, ContentBlockedError
from genmedia.backends.gemini import GeminiImageBackend


class TestGeminiImageBackend:
    def setup_method(self):
        self.client = MagicMock()
        self.backend = GeminiImageBackend(client=self.client)

    def test_build_request_basic(self):
        config = MediaConfig(prompt="a cat", model="gemini-3.1-flash-image-preview")
        req = self.backend.build_request(config)
        assert req["model"] == "gemini-3.1-flash-image-preview"
        assert req["config"]["response_modalities"] == ["IMAGE"]

    def test_build_request_with_aspect_and_size(self):
        config = MediaConfig(prompt="a cat", model="gemini-3.1-flash-image-preview", aspect_ratio="16:9", image_size="4K")
        req = self.backend.build_request(config)
        assert req["config"]["image_config"]["aspect_ratio"] == "16:9"
        assert req["config"]["image_config"]["image_size"] == "4K"

    def test_build_request_edit_mode(self):
        config = MediaConfig(prompt="remove background", model="gemini-3.1-flash-image-preview", input_image=b"fake-image-bytes", input_image_mime="image/png")
        req = self.backend.build_request(config)
        assert req["config"]["response_modalities"] == ["TEXT", "IMAGE"]
        assert len(req["contents"]) == 2

    def test_generate_success(self):
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"fake-png-bytes"
        mock_part.inline_data.mime_type = "image/png"
        mock_part.text = None

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.candidates[0].content.parts = [mock_part]

        self.client.models.generate_content.return_value = mock_response

        config = MediaConfig(prompt="a cat", model="gemini-3.1-flash-image-preview")
        results = self.backend.generate(config)
        assert len(results) == 1
        assert results[0].data == b"fake-png-bytes"
        assert results[0].mime_type == "image/png"

    def test_generate_content_blocked_prompt_level(self):
        mock_response = MagicMock()
        mock_response.prompt_feedback = MagicMock()
        mock_response.prompt_feedback.block_reason = "IMAGE_SAFETY"
        mock_response.candidates = []

        self.client.models.generate_content.return_value = mock_response

        config = MediaConfig(prompt="bad prompt", model="gemini-3.1-flash-image-preview")
        with pytest.raises(ContentBlockedError) as exc_info:
            self.backend.generate(config)
        assert exc_info.value.block_reason == "IMAGE_SAFETY"

    def test_generate_content_blocked_response_level(self):
        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = "SAFETY"
        mock_response.candidates[0].content.parts = []

        self.client.models.generate_content.return_value = mock_response

        config = MediaConfig(prompt="borderline", model="gemini-3.1-flash-image-preview")
        with pytest.raises(ContentBlockedError):
            self.backend.generate(config)

    def test_generate_count_multiple(self):
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"image-bytes"
        mock_part.inline_data.mime_type = "image/png"
        mock_part.text = None

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.candidates[0].content.parts = [mock_part]

        self.client.models.generate_content.return_value = mock_response

        config = MediaConfig(prompt="a cat", model="gemini-3.1-flash-image-preview", count=3)
        results = self.backend.generate(config)
        assert len(results) == 3
        assert self.client.models.generate_content.call_count == 3

    def test_image_size_normalized_uppercase(self):
        config = MediaConfig(prompt="a cat", model="gemini-3.1-flash-image-preview", image_size="4k")
        req = self.backend.build_request(config)
        assert req["config"]["image_config"]["image_size"] == "4K"


from genmedia.backends.imagen import ImagenBackend


class TestImagenBackend:
    def setup_method(self):
        self.client = MagicMock()
        self.backend = ImagenBackend(client=self.client)

    def test_build_request_basic(self):
        config = MediaConfig(prompt="a lake", model="imagen-4.0-generate-001")
        req = self.backend.build_request(config)
        assert req["model"] == "imagen-4.0-generate-001"
        assert req["config"]["number_of_images"] == 1

    def test_build_request_with_count(self):
        config = MediaConfig(prompt="a lake", model="imagen-4.0-generate-001", count=3)
        req = self.backend.build_request(config)
        assert req["config"]["number_of_images"] == 3

    def test_build_request_with_aspect(self):
        config = MediaConfig(prompt="a lake", model="imagen-4.0-generate-001", aspect_ratio="16:9")
        req = self.backend.build_request(config)
        assert req["config"]["aspect_ratio"] == "16:9"

    def test_generate_success(self):
        mock_image = MagicMock()
        mock_image.image.image_bytes = b"imagen-bytes"

        mock_response = MagicMock()
        mock_response.generated_images = [mock_image]

        self.client.models.generate_images.return_value = mock_response

        config = MediaConfig(prompt="a lake", model="imagen-4.0-generate-001")
        results = self.backend.generate(config)
        assert len(results) == 1
        assert results[0].data == b"imagen-bytes"

    def test_generate_uses_native_count(self):
        mock_image = MagicMock()
        mock_image.image.image_bytes = b"imagen-bytes"

        mock_response = MagicMock()
        mock_response.generated_images = [mock_image, mock_image]

        self.client.models.generate_images.return_value = mock_response

        config = MediaConfig(prompt="a lake", model="imagen-4.0-generate-001", count=2)
        results = self.backend.generate(config)
        assert len(results) == 2
        assert self.client.models.generate_images.call_count == 1


from genmedia.backends.veo import VeoBackend


class TestVeoBackend:
    def setup_method(self):
        self.client = MagicMock()
        self.backend = VeoBackend(client=self.client)

    def test_build_request_basic(self):
        config = MediaConfig(prompt="a sunset", model="veo-3.0-generate-001")
        req = self.backend.build_request(config)
        assert req["model"] == "veo-3.0-generate-001"
        assert req["config"]["number_of_videos"] == 1

    def test_build_request_with_duration(self):
        config = MediaConfig(prompt="a sunset", model="veo-3.0-generate-001", duration_seconds=8)
        req = self.backend.build_request(config)
        assert req["config"]["duration_seconds"] == 8

    def test_build_request_with_aspect(self):
        config = MediaConfig(prompt="a sunset", model="veo-3.0-generate-001", aspect_ratio="9:16")
        req = self.backend.build_request(config)
        assert req["config"]["aspect_ratio"] == "9:16"

    def test_generate_success(self):
        mock_video = MagicMock()
        mock_video.video = MagicMock()

        mock_operation = MagicMock()
        mock_operation.done = True
        mock_operation.result.generated_videos = [mock_video]

        self.client.models.generate_videos.return_value = mock_operation
        self.client.files.download.return_value = b"video-bytes"

        config = MediaConfig(prompt="a sunset", model="veo-3.0-generate-001")
        results = self.backend.generate(config)
        assert len(results) == 1
        assert results[0].data == b"video-bytes"
        assert results[0].mime_type == "video/mp4"

    def test_generate_polls_until_done(self):
        mock_video = MagicMock()
        mock_video.video = MagicMock()

        pending_op = MagicMock()
        pending_op.done = False

        done_op = MagicMock()
        done_op.done = True
        done_op.result.generated_videos = [mock_video]

        self.client.models.generate_videos.return_value = pending_op
        self.client.operations.get.return_value = done_op
        self.client.files.download.return_value = b"video-bytes"

        config = MediaConfig(prompt="a sunset", model="veo-3.0-generate-001")
        with patch("genmedia.backends.veo.time.sleep"):
            results = self.backend.generate(config)

        assert len(results) == 1
        self.client.operations.get.assert_called_once()

    def test_generate_keyboard_interrupt(self):
        pending_op = MagicMock()
        pending_op.done = False

        self.client.models.generate_videos.return_value = pending_op
        self.client.operations.get.side_effect = KeyboardInterrupt()

        config = MediaConfig(prompt="a sunset", model="veo-3.0-generate-001")
        with patch("genmedia.backends.veo.time.sleep"):
            with pytest.raises(KeyboardInterrupt):
                self.backend.generate(config)

    def test_generate_no_videos_raises_content_blocked(self):
        mock_operation = MagicMock()
        mock_operation.done = True
        mock_operation.result.generated_videos = None
        mock_operation.result.rai_media_filtered_reasons = None

        self.client.models.generate_videos.return_value = mock_operation

        config = MediaConfig(prompt="bad prompt", model="veo-3.0-generate-001")
        with pytest.raises(ContentBlockedError):
            self.backend.generate(config)

    def test_generate_empty_videos_list_raises_content_blocked(self):
        mock_operation = MagicMock()
        mock_operation.done = True
        mock_operation.result.generated_videos = []
        mock_operation.result.rai_media_filtered_reasons = ["SAFETY"]

        self.client.models.generate_videos.return_value = mock_operation

        config = MediaConfig(prompt="bad prompt", model="veo-3.0-generate-001")
        with pytest.raises(ContentBlockedError) as exc_info:
            self.backend.generate(config)
        assert "SAFETY" in str(exc_info.value)
