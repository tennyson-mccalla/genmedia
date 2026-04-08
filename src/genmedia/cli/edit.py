import json
import mimetypes
import os
import sys
import time

import click
from google import genai

from genmedia.backends.base import ContentBlockedError, MediaConfig
from genmedia.backends.gemini import GeminiImageBackend
from genmedia.models import get_default_model
from genmedia.output import (
    format_dry_run,
    format_error,
    format_success,
    write_media_files,
)
from genmedia.retry import NonRetryableError, RetryableError, RetryWrapper
from genmedia.validation import validate_config


@click.command()
@click.argument("prompt")
@click.option("-i", "--image", "input_images", multiple=True, required=True, type=click.Path(exists=True), help="Input image (repeatable, up to 14)")
@click.option("--model", "-m", default=None, help="Model ID")
@click.option("--output", "-o", default=None, help="Output file path (use - for stdout)")
@click.option("--output-dir", "-d", default=None, help="Output directory")
@click.option("--count", "-n", default=1, type=int, help="Number of variations")
@click.option("--aspect", "-a", default=None, help="Override aspect ratio")
@click.option("--size", "-s", default=None, help="Override image size")
@click.option("--format", "-f", "output_format", default="png", help="Output format: png, jpg, webp")
@click.option("--pretty", is_flag=True, help="Human-friendly output")
@click.option("--dry-run", is_flag=True, help="Show request without calling API")
def edit(prompt, input_images, model, output, output_dir, count, aspect, size, output_format, pretty, dry_run):
    """Edit/compose images. Pass one or more input images via -i/--image."""
    model = model or get_default_model("edit")

    if len(input_images) > 14:
        _exit_error("validation_error", f"At most 14 input images supported, got {len(input_images)}", exit_code=2, pretty=pretty)

    errors = validate_config(
        subcommand="edit",
        prompt=prompt,
        aspect_ratio=aspect,
        image_size=size,
        duration_seconds=None,
        output_format=output_format,
        count=count,
        model=model,
        input_image=input_images[0],
    )

    loaded: list[tuple[bytes, str]] = []
    for path in input_images:
        data = open(path, "rb").read()
        mime = mimetypes.guess_type(path)[0] or "image/png"
        loaded.append((data, mime))

    if dry_run:
        backend = GeminiImageBackend(client=None)
        config = MediaConfig(
            prompt=prompt,
            model=model,
            aspect_ratio=aspect,
            image_size=size,
            output_format=output_format,
            count=count,
            input_images=loaded,
        )
        req = backend.build_request(config)

        click.echo(format_dry_run(
            backend="GeminiImageBackend",
            sdk_method="client.models.generate_content",
            model=model,
            config=req["config"],
            validation_errors=errors,
        ))
        sys.exit(0)

    if errors:
        _exit_error("validation_error", "; ".join(errors), exit_code=2, pretty=pretty)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    backend = GeminiImageBackend(client=client)

    config = MediaConfig(
        prompt=prompt,
        model=model,
        aspect_ratio=aspect,
        image_size=size,
        output_format=output_format,
        count=count,
        input_images=loaded,
    )

    retry = RetryWrapper()
    start = time.monotonic()

    try:
        results = retry.execute(lambda: backend.generate(config))
    except ContentBlockedError as e:
        elapsed = time.monotonic() - start
        _exit_error("content_blocked", str(e), elapsed_seconds=elapsed, block_reason=e.block_reason, exit_code=1, pretty=pretty)
    except RetryableError as e:
        elapsed = time.monotonic() - start
        error_type = "rate_limited" if getattr(e, "status_code", None) == 429 else "server_error"
        _exit_error(error_type, str(e), retries_attempted=retry.attempts, elapsed_seconds=elapsed, exit_code=1, pretty=pretty)
    except NonRetryableError as e:
        elapsed = time.monotonic() - start
        _exit_error("api_error", str(e), elapsed_seconds=elapsed, exit_code=1, pretty=pretty)

    elapsed = time.monotonic() - start

    try:
        written = write_media_files(
            results=results,
            output=output,
            output_dir=output_dir,
            output_format=output_format,
        )
    except OSError as e:
        _exit_error("file_error", str(e), exit_code=3, pretty=pretty)

    request_info = {"prompt": prompt, "input_images": list(input_images)}
    if aspect:
        request_info["aspect_ratio"] = aspect
    if size:
        request_info["image_size"] = size

    if pretty:
        from genmedia.output import format_pretty_success
        click.echo(format_pretty_success(
            files=written,
            model=model,
            elapsed_seconds=elapsed,
        ))
    else:
        click.echo(format_success(
            files=written,
            model=model,
            elapsed_seconds=elapsed,
            request=request_info,
        ))
    sys.exit(0)


def _exit_error(error: str, message: str, exit_code: int = 1, pretty: bool = False, **extra):
    if pretty:
        from genmedia.output import format_pretty_error
        click.echo(format_pretty_error(error=error, message=message), err=True)
    else:
        click.echo(format_error(error=error, message=message, **extra), err=True)
    sys.exit(exit_code)
