import json
import os

from genmedia.backends.base import MediaResult

FORMAT_TO_EXT = {"png": ".png", "jpg": ".jpg", "webp": ".webp", "mp4": ".mp4"}
EXT_TO_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp", ".mp4": "video/mp4"}
DEFAULT_OUTPUT_DIR = "/tmp/genmedia"


def format_success(*, files: list[dict], model: str, elapsed_seconds: float, request: dict) -> str:
    return json.dumps({"status": "success", "files": files, "model": model, "elapsed_seconds": round(elapsed_seconds, 2), "request": request}, indent=2)


def format_error(*, error: str, message: str, retries_attempted: int = 0, elapsed_seconds: float = 0.0, files: list[dict] | None = None, **extra) -> str:
    payload = {"status": "error", "error": error, "message": message, "retries_attempted": retries_attempted, "elapsed_seconds": round(elapsed_seconds, 2)}
    if files:
        payload["files"] = files
    payload.update(extra)
    return json.dumps(payload, indent=2)


def format_dry_run(*, backend: str, sdk_method: str, model: str, config: dict, validation_errors: list[str]) -> str:
    return json.dumps({"status": "dry_run", "backend": backend, "sdk_method": sdk_method, "model": model, "config": config, "validation_errors": validation_errors}, indent=2)


def format_list_models(models: list[dict]) -> str:
    return json.dumps({"models": models}, indent=2)


def auto_name(*, output_dir: str | None = None, extension: str) -> str:
    directory = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(directory, exist_ok=True)
    counter = 1
    while True:
        name = f"genmedia_{counter:03d}{extension}"
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            return path
        counter += 1


def write_media_files(*, results: list[MediaResult], output: str | None, output_dir: str | None, output_format: str) -> list[dict]:
    ext = FORMAT_TO_EXT.get(output_format, f".{output_format}")
    written: list[dict] = []
    for i, result in enumerate(results):
        if output and len(results) == 1:
            path = output
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        else:
            path = auto_name(output_dir=output_dir, extension=ext)
        with open(path, "wb") as f:
            f.write(result.data)
        entry = {"path": os.path.abspath(path), "mime_type": result.mime_type, "size_bytes": len(result.data)}
        if result.metadata:
            entry.update(result.metadata)
        written.append(entry)
    return written
