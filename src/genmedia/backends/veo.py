import os
import time

from google.genai import types
from google.genai.types import VideoGenerationReferenceImage, VideoGenerationReferenceType

from genmedia.backends.base import Backend, ContentBlockedError, MediaConfig, MediaResult
from genmedia.retry import classify_sdk_error

DEFAULT_POLL_TIMEOUT = 600  # 10 minutes


class VeoBackend(Backend):
    POLL_INTERVAL = 5.0

    def __init__(self, client):
        self.client = client
        self.poll_timeout = float(os.environ.get("GENMEDIA_POLL_TIMEOUT", DEFAULT_POLL_TIMEOUT))

    def build_request(self, config: MediaConfig) -> dict:
        veo_config = {"number_of_videos": config.count}
        if config.aspect_ratio:
            veo_config["aspect_ratio"] = config.aspect_ratio
        if config.duration_seconds:
            veo_config["duration_seconds"] = config.duration_seconds
        if config.resolution:
            veo_config["resolution"] = config.resolution
        if config.enhance_prompt:
            veo_config["enhance_prompt"] = True
        return {"model": config.model, "prompt": config.prompt, "config": veo_config}

    def validate(self, config: MediaConfig) -> list[str]:
        return []

    def generate(self, config: MediaConfig) -> list[MediaResult]:
        req = self.build_request(config)

        veo_config = dict(req["config"])

        use_reference_images = config.style_ref or config.asset_refs

        if not use_reference_images and config.last_frame_image:
            veo_config["last_frame"] = types.Image(
                image_bytes=config.last_frame_image,
                mime_type=config.last_frame_mime or "image/png",
            )

        if config.style_ref:
            veo_config["reference_images"] = [
                VideoGenerationReferenceImage(
                    image=types.Image(image_bytes=config.style_ref, mime_type=config.style_ref_mime or "image/png"),
                    reference_type=VideoGenerationReferenceType.STYLE,
                )
            ]
        elif config.asset_refs:
            veo_config["reference_images"] = [
                VideoGenerationReferenceImage(
                    image=types.Image(image_bytes=data, mime_type=mime),
                    reference_type=VideoGenerationReferenceType.ASSET,
                )
                for data, mime in config.asset_refs
            ]

        kwargs = {
            "model": req["model"],
            "prompt": req["prompt"],
            "config": types.GenerateVideosConfig(**veo_config),
        }

        if config.input_image and not use_reference_images:
            kwargs["image"] = types.Image(
                image_bytes=config.input_image,
                mime_type=config.input_image_mime or "image/png",
            )

        try:
            operation = self.client.models.generate_videos(**kwargs)
        except Exception as exc:
            raise classify_sdk_error(exc) from exc

        operation = self._poll_operation(operation)

        if not operation.result or not operation.result.generated_videos:
            reason = None
            if operation.result:
                reason = getattr(operation.result, "rai_media_filtered_reasons", None)
            raise ContentBlockedError(
                f"Video generation produced no output{f': {reason}' if reason else ''}",
                block_reason=str(reason) if reason else "UNKNOWN",
            )

        results = []
        for video in operation.result.generated_videos:
            video_bytes = self.client.files.download(file=video.video)
            results.append(MediaResult(
                data=video_bytes, mime_type="video/mp4",
                metadata={"duration_seconds": config.duration_seconds},
            ))
        return results

    def _poll_operation(self, operation):
        deadline = time.monotonic() + self.poll_timeout
        while not operation.done:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Video generation timed out after {self.poll_timeout:.0f}s")
            time.sleep(self.POLL_INTERVAL)
            operation = self.client.operations.get(operation)
        return operation
