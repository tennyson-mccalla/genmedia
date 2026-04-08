import os
import sys

from genmedia.models import is_known_model

VALID_ASPECT_RATIOS = {
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4",
    "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
}
VIDEO_ASPECT_RATIOS = {"16:9", "9:16"}
VALID_IMAGE_SIZES = {"512", "1K", "2K", "4K"}
VALID_DURATIONS = {4, 6, 8}
VALID_OUTPUT_FORMATS = {"png", "jpg", "webp"}


def validate_config(
    *,
    subcommand: str,
    prompt: str,
    aspect_ratio: str | None,
    image_size: str | None,
    duration_seconds: int | None,
    output_format: str | None,
    count: int,
    model: str,
    input_image: str | None,
) -> list[str]:
    errors: list[str] = []

    if not os.environ.get("GEMINI_API_KEY"):
        errors.append("GEMINI_API_KEY environment variable is not set")
        return errors

    if not prompt or not prompt.strip():
        errors.append("Prompt cannot be empty")

    if aspect_ratio is not None:
        if subcommand == "video" and aspect_ratio not in VIDEO_ASPECT_RATIOS:
            errors.append(
                f"Invalid aspect ratio '{aspect_ratio}' for video. "
                f"Supported: {', '.join(sorted(VIDEO_ASPECT_RATIOS))}"
            )
        elif aspect_ratio not in VALID_ASPECT_RATIOS:
            errors.append(
                f"Invalid aspect ratio '{aspect_ratio}'. "
                f"Supported: {', '.join(sorted(VALID_ASPECT_RATIOS))}"
            )

    if image_size is not None:
        normalized = image_size.upper() if image_size != "512" else image_size
        if normalized not in VALID_IMAGE_SIZES:
            errors.append(
                f"Invalid image size '{image_size}'. "
                f"Supported: {', '.join(sorted(VALID_IMAGE_SIZES))}"
            )
        if model.startswith("imagen"):
            errors.append("image_size is not supported with Imagen models")

    if duration_seconds is not None and duration_seconds not in VALID_DURATIONS:
        errors.append(
            f"Invalid duration {duration_seconds}s. "
            f"Supported: {', '.join(str(d) for d in sorted(VALID_DURATIONS))}"
        )

    if output_format is not None and output_format not in VALID_OUTPUT_FORMATS:
        errors.append(
            f"Invalid output format '{output_format}'. "
            f"Supported: {', '.join(sorted(VALID_OUTPUT_FORMATS))}"
        )

    if count < 1:
        errors.append("Count must be at least 1")

    if not is_known_model(model):
        print(f"Warning: unknown model '{model}'", file=sys.stderr)

    if subcommand == "edit" and input_image is not None:
        if not os.path.isfile(input_image):
            errors.append(f"Input image not found: {input_image}")

    return errors


VEO_RESOLUTIONS = {"720p", "1080p"}
VEO_LAST_FRAME_MODELS = {"veo-3.1-generate-preview"}


def validate_video_extras(
    *,
    resolution: str | None,
    duration_seconds: int | None,
    model: str,
    last_frame: bool,
) -> list[str]:
    errors: list[str] = []
    if resolution is not None and resolution not in VEO_RESOLUTIONS:
        errors.append(
            f"Invalid resolution '{resolution}'. Supported on Gemini API: "
            f"{', '.join(sorted(VEO_RESOLUTIONS))}. (4K is Vertex-only and was removed in v0.3.)"
        )
    if resolution == "1080p" and duration_seconds is not None and duration_seconds != 8:
        errors.append("Resolution 1080p requires --duration 8")
    if last_frame and model not in VEO_LAST_FRAME_MODELS:
        errors.append(
            f"--last-frame is only supported on {', '.join(sorted(VEO_LAST_FRAME_MODELS))}. "
            f"Got '{model}'."
        )
    return errors
