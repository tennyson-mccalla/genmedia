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
