import os
import sys
import time

import click
from google import genai

from genmedia.backends.base import ContentBlockedError, MediaConfig
from genmedia.backends.veo import VeoBackend
from genmedia.models import VIDEO_MODELS, get_default_model
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
@click.option("--count", "-n", default=1, type=int, help="Number of videos")
@click.option("--aspect", "-a", default=None, help="Aspect ratio: 16:9 or 9:16")
@click.option("--duration", default=8, type=int, help="Duration: 4, 6, or 8 seconds")
@click.option("--verbose", "-v", is_flag=True, help="Extra metadata (reserved for future use)")
@click.option("--pretty", is_flag=True, help="Human-friendly output")
@click.option("--dry-run", is_flag=True, help="Show request without calling API")
@click.option("--list-models", is_flag=True, help="List available video models")
def video(prompt, model, output, output_dir, count, aspect, duration, verbose, pretty, dry_run, list_models):
    """Generate video using Veo models."""
    if list_models:
        if pretty:
            from genmedia.output import format_pretty_list_models
            click.echo(format_pretty_list_models(VIDEO_MODELS))
        else:
            click.echo(format_list_models(VIDEO_MODELS))
        sys.exit(0)

    if prompt is None:
        _exit_error("validation_error", "Prompt is required (use --list-models to list models without a prompt)", exit_code=2, pretty=pretty)

    model = model or get_default_model("video")

    if dry_run:
        backend = VeoBackend(client=None)
        config = MediaConfig(
            prompt=prompt,
            model=model,
            aspect_ratio=aspect,
            duration_seconds=duration,
            count=count,
        )

        errors = validate_config(
            subcommand="video",
            prompt=prompt,
            aspect_ratio=aspect,
            image_size=None,
            duration_seconds=duration,
            output_format=None,
            count=count,
            model=model,
            input_image=None,
        )

        req = backend.build_request(config)
        click.echo(format_dry_run(
            backend="VeoBackend",
            sdk_method="client.models.generate_videos",
            model=model,
            config=req["config"],
            validation_errors=errors,
        ))
        sys.exit(0)

    errors = validate_config(
        subcommand="video",
        prompt=prompt,
        aspect_ratio=aspect,
        image_size=None,
        duration_seconds=duration,
        output_format=None,
        count=count,
        model=model,
        input_image=None,
    )
    if errors:
        _exit_error("validation_error", "; ".join(errors), exit_code=2, pretty=pretty)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    backend = VeoBackend(client=client)

    config = MediaConfig(
        prompt=prompt,
        model=model,
        aspect_ratio=aspect,
        duration_seconds=duration,
        count=count,
    )

    retry = RetryWrapper()
    start = time.monotonic()

    try:
        results = retry.execute(lambda: backend.generate(config))
    except KeyboardInterrupt:
        elapsed = time.monotonic() - start
        _exit_error("cancelled", "Polling cancelled. The server-side operation may still be running.", elapsed_seconds=elapsed, exit_code=1, pretty=pretty)
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
            output_format="mp4",
        )
    except OSError as e:
        _exit_error("file_error", str(e), exit_code=3, pretty=pretty)

    request_info = {"prompt": prompt}
    if aspect:
        request_info["aspect_ratio"] = aspect
    if duration:
        request_info["duration_seconds"] = duration

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
