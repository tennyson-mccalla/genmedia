import time

from google.genai import types

from genmedia.backends.base import Backend, ContentBlockedError, MediaConfig, MediaResult
from genmedia.retry import classify_sdk_error


class VeoBackend(Backend):
    POLL_INTERVAL = 5.0

    def __init__(self, client):
        self.client = client

    def build_request(self, config: MediaConfig) -> dict:
        veo_config = {"number_of_videos": config.count}
        if config.aspect_ratio:
            veo_config["aspect_ratio"] = config.aspect_ratio
        if config.duration_seconds:
            veo_config["duration_seconds"] = config.duration_seconds
        return {"model": config.model, "prompt": config.prompt, "config": veo_config}

    def validate(self, config: MediaConfig) -> list[str]:
        return []

    def generate(self, config: MediaConfig) -> list[MediaResult]:
        req = self.build_request(config)
        try:
            operation = self.client.models.generate_videos(
                model=req["model"], prompt=req["prompt"],
                config=types.GenerateVideosConfig(**req["config"]),
            )
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
        while not operation.done:
            time.sleep(self.POLL_INTERVAL)
            operation = self.client.operations.get(operation)
        return operation
