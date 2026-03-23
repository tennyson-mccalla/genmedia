# GenMedia CLI — Project Brief

## Problem

Google's GenAI SDK (`google-genai`) exposes powerful multimodal generation capabilities — images, video, image editing — but there's no good CLI that surfaces them. The closest tool for images, [nanobanana-cli](https://github.com/Factory-AI/nanobanana-cli), hardcodes a single model and ignores critical parameters like aspect ratio and resolution. For video generation (Veo), there's nothing at all. Users who want to generate media from the terminal have to write scripts.

## Goal

A multimodal media generation CLI built on the Google GenAI Python SDK. Subcommand-based architecture covering image generation, video generation, and image editing — with the full parameter surface exposed. Should feel like `curl` for AI media: simple for basic use, powerful when you need it.

## Target Usage

```bash
# Simple image generation
genmedia image "a cat on a skateboard" -o cat.png

# Full control — 4K, specific aspect ratio, specific model
genmedia image "abstract gradient, soft pastels" \
  --model gemini-3.1-flash-image-preview \
  --size 4K \
  --aspect 16:9 \
  --output backgrounds/pastel.png

# Batch image generation
genmedia image "product mockup on marble" --aspect 1:1 --count 3 --output-dir shots/

# Image generation via Imagen 4.0
genmedia image "photorealistic mountain lake at dawn" --model imagen-4.0-generate-001

# Image editing
genmedia edit input.png "remove the background" -o clean.png

# Video generation (Veo)
genmedia video "a cat riding a skateboard down a hill at sunset" \
  --duration 8 \
  --aspect 16:9 \
  --output cat-ride.mp4

# List available models per modality
genmedia image --list-models
genmedia video --list-models
```

## Architecture

Subcommand-based. Each subcommand owns its modality-specific flags; shared options live at the top level.

```
genmedia <subcommand> [prompt] [flags]

Subcommands:
  image    Generate images (Gemini native or Imagen)
  edit     Edit/inpaint an existing image
  video    Generate video (Veo)
```

### Shared flags (all subcommands)

| Flag | Short | Description |
|------|-------|-------------|
| `--model` | `-m` | Model ID (each subcommand has its own default) |
| `--output` | `-o` | Output file path (default: auto-named in cwd) |
| `--output-dir` | `-d` | Output directory for batch |
| `--count` | `-n` | Number of outputs to generate (default: 1) |
| `--aspect` | `-a` | Aspect ratio, e.g. `16:9`, `1:1` |
| `--verbose` | `-v` | Show request details, timing |
| `--dry-run` | | Show what would be sent without calling API |

### `image` subcommand flags

| Flag | Short | Description |
|------|-------|-------------|
| `--size` | `-s` | Image size, e.g. `4K` |
| `--format` | `-f` | Output format: `png`, `jpg`, `webp` |
| `--list-models` | | List available image generation models |

Default model: `gemini-3.1-flash-image-preview`

### `edit` subcommand flags

| Flag | Short | Description |
|------|-------|-------------|
| `--format` | `-f` | Output format: `png`, `jpg`, `webp` |

Takes a positional input image path + prompt. Default model: `gemini-3.1-flash-image-preview`

### `video` subcommand flags

| Flag | Short | Description |
|------|-------|-------------|
| `--duration` | | Video duration in seconds (default: 5) |
| `--list-models` | | List available video generation models |

Default model: `veo-3.0-generate-001`

## Available Models (as of March 2026)

### Image Generation

| Model ID | Notes |
|----------|-------|
| `gemini-2.5-flash-image` | Older, faster |
| `gemini-3-pro-image-preview` | What nanobanana-cli hardcodes |
| `gemini-3.1-flash-image-preview` | **Best quality**, recommended default |
| `imagen-4.0-generate-001` | Imagen — dedicated image gen API, different endpoint |

### Video Generation

| Model ID | Notes |
|----------|-------|
| `veo-3.0-generate-001` | Standard quality, recommended default |
| `veo-3.0-fast-generate-001` | Faster, lower quality |
| `veo-3.1-generate-preview` | Newer preview |
| `veo-3.1-fast-generate-preview` | Newer fast preview |

## API Reference

### Image Generation (Gemini native)

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-3.1-flash-image-preview",
    contents="your prompt here",
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="16:9",   # "1:1", "3:4", "4:3", "9:16", "16:9"
            image_size="4K",       # produces 5504x3072 for 16:9
        ),
    ),
)

for part in response.candidates[0].content.parts:
    if part.inline_data:
        with open("output.png", "wb") as f:
            f.write(part.inline_data.data)
```

### Image Generation (Imagen)

```python
response = client.models.generate_images(
    model="imagen-4.0-generate-001",
    prompt="your prompt here",
    config=types.GenerateImagesConfig(
        aspect_ratio="16:9",
        number_of_images=1,
    ),
)

for image in response.generated_images:
    with open("output.png", "wb") as f:
        f.write(image.image.image_bytes)
```

### Video Generation (Veo)

```python
operation = client.models.generate_videos(
    model="veo-3.0-generate-001",
    prompt="your prompt here",
    config=types.GenerateVideosConfig(
        aspect_ratio="16:9",
        duration_seconds=8,
        number_of_videos=1,
    ),
)

# Video generation is a long-running operation — poll until done
while not operation.done:
    time.sleep(10)
    operation = client.operations.get(operation)

for video in operation.result.generated_videos:
    client.files.download(file=video.video)  # or access video.video.uri
```

## Key Parameters

| Parameter | Values | Notes |
|-----------|--------|-------|
| `aspect_ratio` | `1:1`, `1:4`, `1:8`, `2:3`, `3:2`, `3:4`, `4:1`, `4:3`, `4:5`, `5:4`, `8:1`, `9:16`, `16:9`, `21:9` | Image/edit support all; video only `16:9`, `9:16` |
| `image_size` | `512`, `1K`, `2K`, `4K` | Image only (Gemini, not Imagen). Case-insensitive. |
| `response_modalities` | `["IMAGE"]` or `["TEXT", "IMAGE"]` | `IMAGE` for gen, `TEXT, IMAGE` for edit |
| `duration_seconds` | `4`, `6`, `8` | Video only |

## Environment

- `GEMINI_API_KEY` — required, Gemini API key with billing enabled (free tier is heavily rate-limited)

## Design Decisions (Resolved)

1. **Language**: Python — SDK is mature, `pipx` gives clean installs
2. **Name**: `genmedia`
3. **Scope**: Image generation + image editing + video generation. Audio/live deferred.
4. **Architecture**: Subcommand-based (`image`, `edit`, `video`)

## Design Decisions (Open)

1. **Distribution**: pip/pipx, Homebrew formula, or both
2. **Retry logic**: Gemini 429s frequently — exponential backoff with jitter is required. Configurable max retries or hardcoded sensible default?
3. **`--count` concurrency**: Sequential with progress bar (safer) vs concurrent with semaphore (faster). Leaning sequential for v1.
4. **Auto-naming**: When no `--output` given, use `genmedia_001.png` style with collision avoidance
5. **Video polling UX**: Spinner? Progress bar? Estimated time? Veo jobs can take 30s+.
6. **License**: TBD — possibly FOSS

## Rate Limiting Notes

- Free tier: extremely limited, will 429 constantly on image generation
- Tier 1 (billing enabled): much higher limits but still throttled
- Billing propagation can take hours on existing projects — creating a fresh GCP project with billing enabled from the start works immediately
- Exponential backoff with jitter is non-optional for any serious use

## Prior Art

- **nanobanana-cli** (Factory-AI): Bun-compiled, hardcoded model, no size/aspect flags, ~200 lines of TS
- **Gemini CLI** (Google): General-purpose Gemini CLI, not media-focused
- No dedicated multimodal Gemini media generation CLI exists as of this writing
