import mimetypes
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
from genmedia.validation import validate_config, validate_video_extras


@click.command()
@click.argument("prompt", required=False, default=None)
@click.option("--model", "-m", default=None, help="Model ID")
@click.option("--output", "-o", default=None, help="Output file path (use - for stdout)")
@click.option("--output-dir", "-d", default=None, help="Output directory")
@click.option("--count", "-n", default=1, type=int, help="Number of videos")
@click.option("--aspect", "-a", default=None, help="Aspect ratio: 16:9 or 9:16")
@click.option("--duration", default=8, type=int, help="Duration: 4, 6, or 8 seconds")
@click.option("--image", "-i", "image_path", default=None, help="First frame image for image-to-video (use - for stdin)")
@click.option("--last-frame", default=None, type=click.Path(exists=True), help="Last frame image for frame interpolation (requires veo-3.1-generate-preview)")
@click.option("--resolution", "-r", default=None, type=click.Choice(["720p", "1080p"], case_sensitive=False), help="Video resolution: 720p or 1080p (1080p requires --duration 8)")
@click.option("--enhance-prompt", is_flag=True, help="Let Veo rewrite your prompt for more cinematic results")
@click.option("--negative-prompt", default=None, help="Things to avoid in the video (e.g. 'blurry, low quality')")
@click.option("--style-ref", default=None, type=click.Path(exists=True), help="Style reference image for visual style conditioning")
@click.option("--asset-ref", multiple=True, type=click.Path(exists=True), help="Asset reference image (up to 3, for character/object consistency)")
@click.option("--pretty", is_flag=True, help="Human-friendly output")
@click.option("--dry-run", is_flag=True, help="Show request without calling API")
@click.option("--list-models", is_flag=True, help="List available video models")
def video(prompt, model, output, output_dir, count, aspect, duration, image_path, last_frame, resolution, enhance_prompt, negative_prompt, style_ref, asset_ref, pretty, dry_run, list_models):
    """Generate video using Veo models."""
    if list_models:
        if pretty:
            from genmedia.output import format_pretty_list_models
            click.echo(format_pretty_list_models(VIDEO_MODELS))
        else:
            click.echo(format_list_models(VIDEO_MODELS))
        sys.exit(0)

    # Reference image validation
    if style_ref and asset_ref:
        _exit_error("validation_error", "--style-ref and --asset-ref cannot be used together", exit_code=2, pretty=pretty)

    if (style_ref or asset_ref) and (image_path or last_frame):
        _exit_error("validation_error", "--style-ref/--asset-ref cannot be used with --image or --last-frame", exit_code=2, pretty=pretty)

    if (style_ref or asset_ref) and not prompt:
        _exit_error("validation_error", "--style-ref/--asset-ref require a text prompt", exit_code=2, pretty=pretty)

    if prompt is None and not image_path and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()

    if not prompt and not image_path:
        _exit_error("validation_error", "Prompt or --image is required (use --list-models to list models without a prompt)", exit_code=2, pretty=pretty)

    if last_frame and not image_path:
        _exit_error("validation_error", "--last-frame requires --image (first frame)", exit_code=2, pretty=pretty)

    model = model or get_default_model("video")
    prompt = prompt or ""

    # Load images if provided
    input_image_bytes = None
    input_image_mime = None
    last_frame_bytes = None
    last_frame_mime = None

    if image_path == "-":
        input_image_bytes = sys.stdin.buffer.read()
        input_image_mime = "image/png"
    elif image_path:
        if not os.path.isfile(image_path):
            _exit_error("validation_error", f"Image file not found: {image_path}", exit_code=2, pretty=pretty)
        input_image_bytes = open(image_path, "rb").read()
        input_image_mime = mimetypes.guess_type(image_path)[0] or "image/png"

    if last_frame:
        last_frame_bytes = open(last_frame, "rb").read()
        last_frame_mime = mimetypes.guess_type(last_frame)[0] or "image/png"

    # Load reference images
    style_ref_bytes = None
    style_ref_mime = None
    asset_refs_loaded = None

    if style_ref:
        style_ref_bytes = open(style_ref, "rb").read()
        style_ref_mime = mimetypes.guess_type(style_ref)[0] or "image/png"

    if asset_ref:
        asset_refs_loaded = []
        for path in asset_ref:
            data = open(path, "rb").read()
            mime = mimetypes.guess_type(path)[0] or "image/png"
            asset_refs_loaded.append((data, mime))

    if dry_run:
        backend = VeoBackend(client=None)
        config = MediaConfig(
            prompt=prompt,
            model=model,
            aspect_ratio=aspect,
            duration_seconds=duration,
            count=count,
            input_image=input_image_bytes,
            input_image_mime=input_image_mime,
            last_frame_image=last_frame_bytes,
            last_frame_mime=last_frame_mime,
            resolution=resolution,
            enhance_prompt=enhance_prompt,
            style_ref=style_ref_bytes,
            style_ref_mime=style_ref_mime,
            asset_refs=asset_refs_loaded,
            negative_prompt=negative_prompt,
        )

        errors = validate_config(
            subcommand="video",
            prompt=prompt if prompt else "image-to-video",
            aspect_ratio=aspect,
            image_size=None,
            duration_seconds=duration,
            output_format=None,
            count=count,
            model=model,
            input_image=None,
        )

        req = backend.build_request(config)
        dry_run_config = dict(req["config"])
        if image_path:
            dry_run_config["image"] = image_path
        if last_frame:
            dry_run_config["last_frame"] = last_frame
        if resolution:
            dry_run_config["resolution"] = resolution
        if enhance_prompt:
            dry_run_config["enhance_prompt"] = enhance_prompt
        if negative_prompt:
            dry_run_config["negative_prompt"] = negative_prompt
        if style_ref:
            dry_run_config["style_ref"] = style_ref
        if asset_ref:
            dry_run_config["asset_refs"] = list(asset_ref)

        click.echo(format_dry_run(
            backend="VeoBackend",
            sdk_method="client.models.generate_videos",
            model=model,
            config=dry_run_config,
            validation_errors=errors,
        ))
        sys.exit(0)

    errors = validate_config(
        subcommand="video",
        prompt=prompt if prompt else "image-to-video",
        aspect_ratio=aspect,
        image_size=None,
        duration_seconds=duration,
        output_format=None,
        count=count,
        model=model,
        input_image=None,
    )
    errors += validate_video_extras(
        resolution=resolution,
        duration_seconds=duration,
        model=model,
        last_frame=bool(last_frame),
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
        input_image=input_image_bytes,
        input_image_mime=input_image_mime,
        last_frame_image=last_frame_bytes,
        last_frame_mime=last_frame_mime,
        resolution=resolution,
        enhance_prompt=enhance_prompt,
        style_ref=style_ref_bytes,
        style_ref_mime=style_ref_mime,
        asset_refs=asset_refs_loaded,
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
    except TimeoutError as e:
        elapsed = time.monotonic() - start
        _exit_error("timeout", str(e), elapsed_seconds=elapsed, exit_code=1, pretty=pretty)
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
    if image_path:
        request_info["image"] = image_path
    if last_frame:
        request_info["last_frame"] = last_frame
    if resolution:
        request_info["resolution"] = resolution
    if enhance_prompt:
        request_info["enhance_prompt"] = enhance_prompt
    if negative_prompt:
        request_info["negative_prompt"] = negative_prompt
    if style_ref:
        request_info["style_ref"] = style_ref
    if asset_ref:
        request_info["asset_refs"] = list(asset_ref)

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
