import json
import os
import sys
import time

import click
from google import genai

from genmedia.backends.base import ContentBlockedError, MediaConfig
from genmedia.backends.gemini import GeminiImageBackend
from genmedia.backends.imagen import ImagenBackend
from genmedia.models import IMAGE_MODELS, get_default_model
from genmedia.output import (
    format_dry_run,
    format_error,
    format_list_models,
    format_success,
    write_media_files,
)
from genmedia.retry import NonRetryableError, RetryableError, RetryWrapper
from genmedia.validation import validate_config


@click.command()
@click.argument("prompt", required=False, default=None)
@click.option("--model", "-m", default=None, help="Model ID")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--output-dir", "-d", default=None, help="Output directory")
@click.option("--count", "-n", default=1, type=int, help="Number of images")
@click.option("--aspect", "-a", default=None, help="Aspect ratio (e.g. 16:9)")
@click.option("--size", "-s", default=None, help="Image size: 512, 1K, 2K, 4K")
@click.option("--format", "-f", "output_format", default="png", help="Output format: png, jpg, webp")
@click.option("--verbose", "-v", is_flag=True, help="Extra metadata in output")
@click.option("--pretty", is_flag=True, help="Human-friendly output")
@click.option("--dry-run", is_flag=True, help="Show request without calling API")
@click.option("--list-models", is_flag=True, help="List available models")
@click.option("--json", "json_flag", is_flag=True, hidden=True, help="JSON output (default, no-op)")
def image(prompt, model, output, output_dir, count, aspect, size, output_format, verbose, pretty, dry_run, list_models, json_flag):
    """Generate images using Gemini or Imagen models."""
    if list_models:
        click.echo(format_list_models(IMAGE_MODELS))
        sys.exit(0)

    if prompt is None:
        _exit_error("validation_error", "Prompt is required (use --list-models to list models without a prompt)", exit_code=2)

    model = model or get_default_model("image")
    is_imagen = model.startswith("imagen")

    if dry_run:
        backend_cls = ImagenBackend if is_imagen else GeminiImageBackend
        backend = backend_cls(client=None)
        config = MediaConfig(
            prompt=prompt,
            model=model,
            aspect_ratio=aspect,
            image_size=size,
            output_format=output_format,
            count=count,
        )

        errors = validate_config(
            subcommand="image",
            prompt=prompt,
            aspect_ratio=aspect,
            image_size=size,
            duration_seconds=None,
            output_format=output_format,
            count=count,
            model=model,
            input_image=None,
        )

        req = backend.build_request(config)
        backend_name = "ImagenBackend" if is_imagen else "GeminiImageBackend"
        sdk_method = "client.models.generate_images" if is_imagen else "client.models.generate_content"

        click.echo(format_dry_run(
            backend=backend_name,
            sdk_method=sdk_method,
            model=model,
            config=req["config"],
            validation_errors=errors,
        ))
        sys.exit(0)

    errors = validate_config(
        subcommand="image",
        prompt=prompt,
        aspect_ratio=aspect,
        image_size=size,
        duration_seconds=None,
        output_format=output_format,
        count=count,
        model=model,
        input_image=None,
    )
    if errors:
        _exit_error("validation_error", "; ".join(errors), exit_code=2)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    backend_cls = ImagenBackend if is_imagen else GeminiImageBackend
    backend = backend_cls(client=client)

    config = MediaConfig(
        prompt=prompt,
        model=model,
        aspect_ratio=aspect,
        image_size=size,
        output_format=output_format,
        count=count,
    )

    retry = RetryWrapper()
    start = time.monotonic()

    try:
        results = retry.execute(lambda: backend.generate(config))
    except ContentBlockedError as e:
        elapsed = time.monotonic() - start
        _exit_error("content_blocked", str(e), elapsed_seconds=elapsed, block_reason=e.block_reason, exit_code=1)
    except RetryableError as e:
        elapsed = time.monotonic() - start
        error_type = "rate_limited" if getattr(e, "status_code", None) == 429 else "server_error"
        _exit_error(error_type, str(e), retries_attempted=retry.attempts, elapsed_seconds=elapsed, exit_code=1)
    except NonRetryableError as e:
        elapsed = time.monotonic() - start
        _exit_error("api_error", str(e), elapsed_seconds=elapsed, exit_code=1)

    elapsed = time.monotonic() - start

    try:
        written = write_media_files(
            results=results,
            output=output,
            output_dir=output_dir,
            output_format=output_format,
        )
    except OSError as e:
        _exit_error("file_error", str(e), exit_code=3)

    request_info = {"prompt": prompt}
    if aspect:
        request_info["aspect_ratio"] = aspect
    if size:
        request_info["image_size"] = size

    click.echo(format_success(
        files=written,
        model=model,
        elapsed_seconds=elapsed,
        request=request_info,
    ))
    sys.exit(0)


def _exit_error(error: str, message: str, exit_code: int = 1, **extra):
    click.echo(format_error(error=error, message=message, **extra), err=True)
    sys.exit(exit_code)
