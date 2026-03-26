from concurrent.futures import ThreadPoolExecutor, as_completed

from google.genai import types

from genmedia.backends.base import Backend, ContentBlockedError, MediaConfig, MediaResult
from genmedia.retry import classify_sdk_error


class GeminiImageBackend(Backend):
    def __init__(self, client):
        self.client = client

    def build_request(self, config: MediaConfig) -> dict:
        is_edit = config.input_image is not None

        if is_edit:
            contents = [
                config.prompt,
                types.Part.from_bytes(data=config.input_image, mime_type=config.input_image_mime),
            ]
            modalities = ["TEXT", "IMAGE"]
        else:
            contents = config.prompt
            modalities = ["IMAGE"]

        image_config = {}
        if config.aspect_ratio:
            image_config["aspect_ratio"] = config.aspect_ratio
        if config.image_size:
            normalized = config.image_size.upper() if config.image_size != "512" else "512"
            image_config["image_size"] = normalized

        sdk_config = {"response_modalities": modalities}
        if image_config:
            sdk_config["image_config"] = image_config

        return {"model": config.model, "contents": contents, "config": sdk_config}

    def validate(self, config: MediaConfig) -> list[str]:
        return []

    def _generate_one(self, config: MediaConfig) -> list[MediaResult]:
        """Generate a single image. Returns list of MediaResult from one API call."""
        req = self.build_request(config)

        image_config_obj = None
        if "image_config" in req["config"]:
            image_config_obj = types.ImageConfig(**req["config"]["image_config"])

        try:
            response = self.client.models.generate_content(
                model=req["model"],
                contents=req["contents"],
                config=types.GenerateContentConfig(
                    response_modalities=req["config"]["response_modalities"],
                    image_config=image_config_obj,
                ),
            )
        except Exception as exc:
            raise classify_sdk_error(exc) from exc

        self._check_safety(response)

        if not response.candidates:
            raise ContentBlockedError("No candidates in response", block_reason="UNKNOWN")

        results = []
        parts = getattr(response.candidates[0].content, "parts", None) or []
        for part in parts:
            if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                results.append(
                    MediaResult(
                        data=part.inline_data.data,
                        mime_type=getattr(part.inline_data, "mime_type", "image/png"),
                        metadata={},
                    )
                )
        return results

    def generate(self, config: MediaConfig) -> list[MediaResult]:
        if config.count == 1:
            results = self._generate_one(config)
        else:
            results: list[MediaResult] = []
            max_workers = min(config.count, 3)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(self._generate_one, config) for _ in range(config.count)]
                for future in as_completed(futures):
                    results.extend(future.result())

        if not results:
            raise ContentBlockedError(
                "Generation produced no image output",
                block_reason="UNKNOWN",
            )
        return results

    def _check_safety(self, response) -> None:
        if response.prompt_feedback and getattr(response.prompt_feedback, "block_reason", None):
            reason = str(response.prompt_feedback.block_reason)
            raise ContentBlockedError(f"Prompt blocked by safety filter: {reason}", block_reason=reason)

        if response.candidates:
            finish = getattr(response.candidates[0], "finish_reason", None)
            if finish == "SAFETY" or str(finish) == "SAFETY":
                raise ContentBlockedError("Response blocked by safety filter", block_reason="SAFETY")
