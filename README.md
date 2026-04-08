# genmedia

Multimodal media generation CLI for Google GenAI. Images, video, and image editing from the terminal. Bring your own API key.

## Breaking changes in 0.3.0

- `genmedia edit` now takes the prompt as the only positional argument and one or more `-i/--image` flags: `genmedia edit "merge them" -i a.png -i b.png` (up to 14 images). The old `genmedia edit input.png "prompt"` form and the `-` stdin path are gone.
- `--resolution 4K` removed. The public Gemini API rejects 4K/2160p on every Veo model. Only `720p` and `1080p` are accepted, and `1080p` requires `--duration 8`.
- `--last-frame` is now gated to `veo-3.1-generate-preview` only (the only model that actually supports it).
- `veo-2.0-generate-001` removed from the model list (Vertex-only; genmedia has no Vertex backend).

## New in 0.3.0

- `genmedia video --negative-prompt "blurry, low quality"` — things to avoid.
- `genmedia image` Imagen knobs: `--guidance-scale`, `--person-generation` (`ALLOW_ADULT`/`DONT_ALLOW`), `--compression-quality` (jpg only).
- `genmedia image --size 1K`/`--size 2K` now works on Imagen models (previously incorrectly rejected).
- `genmedia edit` accepts up to 14 input images for composition.

## Examples

| Generate | Edit | Generate |
|----------|------|----------|
| ![cat on skateboard](https://github.com/tennyson-mccalla/genmedia/releases/download/v0.1.1/genmedia_001.jpg) | ![red triangle](https://github.com/tennyson-mccalla/genmedia/releases/download/v0.1.1/genmedia_003.jpg) | ![crystal orb](https://github.com/tennyson-mccalla/genmedia/releases/download/v0.1.1/genmedia_005.jpg) |
| `genmedia image "a cat on a skateboard"` | *before editing* | `genmedia image "a glowing crystal orb floating in darkness"` |
| ![crystal cat](https://github.com/tennyson-mccalla/genmedia/releases/download/v0.1.1/genmedia_006.jpg) | ![blue triangle](https://github.com/tennyson-mccalla/genmedia/releases/download/v0.1.1/genmedia_004.jpg) | |
| `genmedia image "a slightly glowing crystalline cat in the morning light in a forest, sharp, magical realism"` | `genmedia edit red_triangle.jpg "change to blue, yellow gradient background"` | |

https://github.com/user-attachments/assets/46beecf4-a12a-4983-be7e-5ce516b8b803

## Install

```bash
# uv (fastest)
uv tool install genmedia

# or run without installing
uvx genmedia image "a cat on a skateboard" --pretty

# pipx
pipx install genmedia

# pip
pip install genmedia
```

Requires Python 3.11+.

## Setup

Get a [Gemini API key](https://aistudio.google.com/apikey) and set it:

```bash
export GEMINI_API_KEY="your-key-here"
```

## Usage

### Generate images

```bash
# Simple
genmedia image "a cat on a skateboard" --pretty

# Full control
genmedia image "abstract gradient, soft pastels" \
  --model gemini-3.1-flash-image-preview \
  --size 2K \
  --aspect 16:9 \
  --output backgrounds/pastel.png

# Batch
genmedia image "product mockup on marble" --count 3 --output-dir shots/

# Using Imagen
genmedia image "photorealistic mountain lake" --model imagen-4.0-generate-001
```

### Edit images

```bash
genmedia edit "remove the background" -i input.png --pretty
genmedia edit "make the sky more dramatic" -i photo.jpg -o edited.jpg

# Compose multiple images (up to 14)
genmedia edit "place the character in the forest" -i character.png -i forest.jpg
```

### Generate video

```bash
# Text-to-video
genmedia video "a cat riding a skateboard downhill at sunset" --duration 8 --pretty

# Image-to-video (animate a still image)
genmedia video "the scene comes to life" --image keyframe.jpg --pretty

# Style reference (apply visual style from an image)
genmedia video "a bustling city street" --style-ref painting.jpg

# Asset reference (character/object consistency, up to 3 images)
genmedia video "the character walks through a forest" --asset-ref character.png
```

> **Note:** `--last-frame` (frame interpolation) is only supported by `veo-3.1-generate-preview`. `--resolution 1080p` requires `--duration 8`. Use `--negative-prompt "blurry, low quality"` to steer Veo away from artifacts.

### List models

```bash
genmedia image --list-models
genmedia video --list-models
```

### Piping

```bash
# Pipe prompt from stdin
echo "a neon cityscape at night" | genmedia image --pretty

# Write binary to stdout
genmedia image "a logo" --output - > logo.png

# Read image from stdin for video
cat keyframe.png | genmedia video "animate this" --image - --pretty

# (Note: `genmedia edit` no longer accepts stdin — pass files with `-i`.)
```

### Dry run

```bash
genmedia image "test" --dry-run   # shows request payload without calling API
```

## Output

JSON by default (for AI agents and scripts):

```json
{
  "status": "success",
  "files": [
    {
      "path": "/tmp/genmedia/genmedia_001.jpg",
      "mime_type": "image/jpeg",
      "size_bytes": 534182
    }
  ],
  "model": "gemini-3.1-flash-image-preview",
  "elapsed_seconds": 14.3,
  "request": {
    "prompt": "a cat on a skateboard",
    "aspect_ratio": "16:9"
  }
}
```

Use `--pretty` for human-friendly output:

```
Saved to /tmp/genmedia/genmedia_001.jpg (521.7 KB)
Model: gemini-3.1-flash-image-preview | Time: 14.3s
```

Errors go to stderr as JSON with distinct exit codes:
- `0` — success
- `1` — API error (rate limit, server error, content blocked)
- `2` — validation error (bad params, missing API key)
- `3` — file I/O error

## Models

### Image generation

| Model | Notes |
|-------|-------|
| `gemini-3.1-flash-image-preview` | Default. Best quality. Multi-image edit/composition (up to 14). |
| `gemini-3-pro-image-preview` | Previous generation. |
| `gemini-2.5-flash-image` | Older, faster. |
| `imagen-4.0-generate-001` | Imagen. Different API, good for photorealism. Only model that supports `--guidance-scale`, `--person-generation`, and `--compression-quality` (jpg only). |

Knob compatibility:

| Flag | Applies to |
|------|------------|
| `--guidance-scale` | Imagen models only |
| `--person-generation` | Imagen models only (`ALLOW_ADULT`, `DONT_ALLOW`; `ALLOW_ALL` is Vertex-only) |
| `--compression-quality` | Imagen models only, requires `--format jpg` |
| `--size` | All image models (Imagen + Gemini) — `512`, `1K`, `2K` |
| `-i` (multi) | `gemini-3.1-flash-image-preview` for `genmedia edit` composition |

### Video generation

| Model | Notes |
|-------|-------|
| `veo-3.0-generate-001` | Default. Standard quality. |
| `veo-3.0-fast-generate-001` | Faster, lower quality. |
| `veo-3.1-generate-preview` | Newer preview. Only model that supports `--last-frame` (frame interpolation). |
| `veo-3.1-fast-generate-preview` | Newer fast preview. Does not support `--last-frame` (API rejects with 400 — verified 2026-04-07). |

Knob compatibility:

| Flag | Applies to |
|------|------------|
| `--negative-prompt` | All Veo models |
| `--last-frame` | `veo-3.1-generate-preview` only |
| `--resolution 1080p` | All Veo models, requires `--duration 8` |
| `--resolution 720p` | All Veo models |
| `--style-ref`, `--asset-ref` | All Veo models |

## For AI agents

genmedia is designed to be called by AI agents. JSON output by default, structured errors on stderr, distinct exit codes for branching. No interactive prompts, no spinners unless `--pretty` is set.

```python
import subprocess, json

result = subprocess.run(
    ["genmedia", "image", "a logo for my app", "--aspect", "1:1"],
    capture_output=True, text=True
)

if result.returncode == 0:
    data = json.loads(result.stdout)
    image_path = data["files"][0]["path"]
```

## License

MIT
