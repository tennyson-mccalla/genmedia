# genmedia

Multimodal media generation CLI for Google GenAI. Images, video, and image editing from the terminal. Bring your own API key.

## Examples

| Generate | Edit | Generate |
|----------|------|----------|
| ![cat on skateboard](https://github.com/tennyson-mccalla/genmedia/releases/download/v0.1.1/genmedia_001.jpg) | ![red triangle](https://github.com/tennyson-mccalla/genmedia/releases/download/v0.1.1/genmedia_003.jpg) | ![crystal orb](https://github.com/tennyson-mccalla/genmedia/releases/download/v0.1.1/genmedia_005.jpg) |
| `genmedia image "a cat on a skateboard"` | *before editing* | `genmedia image "a glowing crystal orb floating in darkness"` |
| ![crystal cat](https://github.com/tennyson-mccalla/genmedia/releases/download/v0.1.1/genmedia_006.jpg) | ![blue triangle](https://github.com/tennyson-mccalla/genmedia/releases/download/v0.1.1/genmedia_004.jpg) | |
| `genmedia image "crystal cat in a forest"` | `genmedia edit red_triangle.jpg "change to blue, yellow gradient background"` | |

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
  --size 4K \
  --aspect 16:9 \
  --output backgrounds/pastel.png

# Batch
genmedia image "product mockup on marble" --count 3 --output-dir shots/

# Using Imagen
genmedia image "photorealistic mountain lake" --model imagen-4.0-generate-001
```

### Edit images

```bash
genmedia edit input.png "remove the background" --pretty
genmedia edit photo.jpg "make the sky more dramatic" -o edited.jpg
```

### Generate video

```bash
# Text-to-video
genmedia video "a cat riding a skateboard downhill at sunset" --duration 8 --pretty

# Image-to-video (animate a still image)
genmedia video "the scene comes to life" --image keyframe.jpg --pretty

# Frame interpolation (morph between two images)
genmedia video "smooth transition" --image start.jpg --last-frame end.jpg
```

### List models

```bash
genmedia image --list-models
genmedia video --list-models
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
| `gemini-3.1-flash-image-preview` | Default. Best quality. |
| `gemini-3-pro-image-preview` | Previous generation. |
| `gemini-2.5-flash-image` | Older, faster. |
| `imagen-4.0-generate-001` | Imagen. Different API, good for photorealism. |

### Video generation

| Model | Notes |
|-------|-------|
| `veo-3.0-generate-001` | Default. Standard quality. |
| `veo-3.0-fast-generate-001` | Faster, lower quality. |
| `veo-3.1-generate-preview` | Newer preview. |
| `veo-3.1-fast-generate-preview` | Newer fast preview. |

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
