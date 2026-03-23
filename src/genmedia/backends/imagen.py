from google.genai import types

from genmedia.backends.base import Backend, MediaConfig, MediaResult
from genmedia.retry import classify_sdk_error


class ImagenBackend(Backend):
    def __init__(self, client):
        self.client = client

    def build_request(self, config: MediaConfig) -> dict:
        imagen_config = {"number_of_images": config.count}
        if config.aspect_ratio:
            imagen_config["aspect_ratio"] = config.aspect_ratio
        if config.output_format:
            mime_map = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}
            imagen_config["output_mime_type"] = mime_map.get(config.output_format, "image/png")
        return {"model": config.model, "prompt": config.prompt, "config": imagen_config}

    def validate(self, config: MediaConfig) -> list[str]:
        errors = []
        if config.image_size:
            errors.append("image_size is not supported with Imagen models")
        return errors

    def generate(self, config: MediaConfig) -> list[MediaResult]:
        req = self.build_request(config)
        try:
            response = self.client.models.generate_images(
                model=req["model"], prompt=req["prompt"],
                config=types.GenerateImagesConfig(**req["config"]),
            )
        except Exception as exc:
            raise classify_sdk_error(exc) from exc
        results = []
        for image in response.generated_images:
            results.append(MediaResult(data=image.image.image_bytes, mime_type="image/png", metadata={}))
        return results
