# GenMedia CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multimodal media generation CLI (`genmedia`) on the Google GenAI Python SDK with image, edit, and video subcommands.

**Architecture:** Subcommand-based CLI (Click) with a backend abstraction layer. Three backends (GeminiImage, Imagen, Veo) implement a shared interface. Shared infrastructure handles retry, validation, output formatting, and auto-naming. JSON output by default for AI consumers.

**Tech Stack:** Python 3.11+, Click, google-genai SDK, pytest

**Spec:** `docs/superpowers/specs/2026-03-23-genmedia-cli-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Package metadata, dependencies, entry point |
| `src/genmedia/__init__.py` | Package version |
| `src/genmedia/models.py` | Known model registry (hardcoded model lists) |
| `src/genmedia/validation.py` | Parameter validation (aspect ratio, size, duration, prompt, API key) |
| `src/genmedia/retry.py` | Exponential backoff with jitter, RetryableError |
| `src/genmedia/output.py` | JSON + pretty output formatting, auto-naming, file writing |
| `src/genmedia/backends/base.py` | Backend ABC, MediaResult dataclass, MediaConfig dataclass, ContentBlockedError |
| `src/genmedia/backends/gemini.py` | GeminiImageBackend (generate_content for image + edit) |
| `src/genmedia/backends/imagen.py` | ImagenBackend (generate_images) |
| `src/genmedia/backends/veo.py` | VeoBackend (generate_videos + polling) |
| `src/genmedia/backends/__init__.py` | Backend exports |
| `src/genmedia/cli/__init__.py` | CLI exports |
| `src/genmedia/cli/main.py` | Click group, shared options decorator |
| `src/genmedia/cli/image.py` | image subcommand |
| `src/genmedia/cli/edit.py` | edit subcommand |
| `src/genmedia/cli/video.py` | video subcommand |
| `tests/unit/test_validation.py` | Validation unit tests |
| `tests/unit/test_retry.py` | Retry unit tests |
| `tests/unit/test_output.py` | Output formatting + auto-naming tests |
| `tests/unit/test_backends.py` | Backend request building + response handling tests |
| `tests/unit/test_cli.py` | CLI parsing + integration between layers |
| `tests/integration/conftest.py` | GENMEDIA_TEST_LIVE gate |
| `tests/integration/test_gemini_image.py` | Live Gemini image gen test |
| `tests/integration/test_imagen.py` | Live Imagen test |
| `tests/integration/test_veo.py` | Live Veo video test |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/genmedia/__init__.py`
- Create: `src/genmedia/backends/__init__.py`
- Create: `src/genmedia/cli/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.backends"

[project]
name = "genmedia"
version = "0.1.0"
description = "Multimodal media generation CLI for Google GenAI"
requires-python = ">=3.11"
dependencies = [
    "google-genai>=1.0.0",
    "click>=8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.0",
]

[project.scripts]
genmedia = "genmedia.cli.main:cli"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package init files**

`src/genmedia/__init__.py`:
```python
__version__ = "0.1.0"
```

`src/genmedia/backends/__init__.py`:
```python
```

`src/genmedia/cli/__init__.py`:
```python
```

- [ ] **Step 3: Install the package in dev mode**

Run: `cd /Users/Tennyson/genmedia && pip install -e ".[dev]"`
Expected: Successful install with genmedia entry point registered

- [ ] **Step 4: Verify pytest runs with no tests**

Run: `cd /Users/Tennyson/genmedia && pytest --co -q`
Expected: "no tests ran" or similar (no errors)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/
git commit -m "scaffold: project structure with pyproject.toml and package init"
```

---

### Task 2: Models Registry

**Files:**
- Create: `src/genmedia/models.py`
- Create: `tests/unit/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_models.py`:
```python
from genmedia.models import IMAGE_MODELS, VIDEO_MODELS, get_default_model, is_known_model


def test_image_models_contains_default():
    default = get_default_model("image")
    assert default == "gemini-3.1-flash-image-preview"
    assert any(m["id"] == default for m in IMAGE_MODELS)


def test_video_models_contains_default():
    default = get_default_model("video")
    assert default == "veo-3.0-generate-001"
    assert any(m["id"] == default for m in VIDEO_MODELS)


def test_edit_default_same_as_image():
    assert get_default_model("edit") == get_default_model("image")


def test_is_known_model():
    assert is_known_model("gemini-3.1-flash-image-preview") is True
    assert is_known_model("imagen-4.0-generate-001") is True
    assert is_known_model("veo-3.0-generate-001") is True
    assert is_known_model("nonexistent-model") is False


def test_model_entries_have_required_fields():
    for model in IMAGE_MODELS + VIDEO_MODELS:
        assert "id" in model
        assert "notes" in model
        assert "default" in model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_models.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write implementation**

`src/genmedia/models.py`:
```python
IMAGE_MODELS = [
    {"id": "gemini-3.1-flash-image-preview", "default": True, "notes": "Best quality, recommended"},
    {"id": "gemini-3-pro-image-preview", "default": False, "notes": "Previous generation"},
    {"id": "gemini-2.5-flash-image", "default": False, "notes": "Older, faster"},
    {"id": "imagen-4.0-generate-001", "default": False, "notes": "Imagen — different API endpoint"},
]

VIDEO_MODELS = [
    {"id": "veo-3.0-generate-001", "default": True, "notes": "Standard quality"},
    {"id": "veo-3.0-fast-generate-001", "default": False, "notes": "Faster, lower quality"},
    {"id": "veo-3.1-generate-preview", "default": False, "notes": "Newer preview"},
    {"id": "veo-3.1-fast-generate-preview", "default": False, "notes": "Newer fast preview"},
]

_ALL_MODELS = {m["id"] for m in IMAGE_MODELS + VIDEO_MODELS}

_DEFAULTS = {
    "image": "gemini-3.1-flash-image-preview",
    "edit": "gemini-3.1-flash-image-preview",
    "video": "veo-3.0-generate-001",
}


def get_default_model(subcommand: str) -> str:
    return _DEFAULTS[subcommand]


def is_known_model(model_id: str) -> bool:
    return model_id in _ALL_MODELS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_models.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/genmedia/models.py tests/unit/test_models.py
git commit -m "feat: add hardcoded model registry"
```

---

### Task 3: Validation Module

**Files:**
- Create: `src/genmedia/validation.py`
- Create: `tests/unit/test_validation.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_validation.py`:
```python
import os
from unittest.mock import patch

from genmedia.validation import validate_config


def test_missing_api_key():
    with patch.dict(os.environ, {}, clear=True):
        errors = validate_config(
            subcommand="image",
            prompt="hello",
            aspect_ratio=None,
            image_size=None,
            duration_seconds=None,
            output_format="png",
            count=1,
            model="gemini-3.1-flash-image-preview",
            input_image=None,
        )
    assert any("GEMINI_API_KEY" in e for e in errors)


def test_empty_prompt():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image",
            prompt="",
            aspect_ratio=None,
            image_size=None,
            duration_seconds=None,
            output_format="png",
            count=1,
            model="gemini-3.1-flash-image-preview",
            input_image=None,
        )
    assert any("empty" in e.lower() for e in errors)


def test_whitespace_prompt():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image",
            prompt="   ",
            aspect_ratio=None,
            image_size=None,
            duration_seconds=None,
            output_format="png",
            count=1,
            model="gemini-3.1-flash-image-preview",
            input_image=None,
        )
    assert any("empty" in e.lower() for e in errors)


def test_valid_aspect_ratios():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        for ratio in ["1:1", "16:9", "9:16", "3:4", "4:3", "21:9"]:
            errors = validate_config(
                subcommand="image",
                prompt="test",
                aspect_ratio=ratio,
                image_size=None,
                duration_seconds=None,
                output_format="png",
                count=1,
                model="gemini-3.1-flash-image-preview",
                input_image=None,
            )
            assert not errors, f"Unexpected error for {ratio}: {errors}"


def test_invalid_aspect_ratio():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image",
            prompt="test",
            aspect_ratio="7:3",
            image_size=None,
            duration_seconds=None,
            output_format="png",
            count=1,
            model="gemini-3.1-flash-image-preview",
            input_image=None,
        )
    assert any("aspect" in e.lower() for e in errors)


def test_video_aspect_ratio_restricted():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="video",
            prompt="test",
            aspect_ratio="4:3",
            image_size=None,
            duration_seconds=8,
            output_format=None,
            count=1,
            model="veo-3.0-generate-001",
            input_image=None,
        )
    assert any("video" in e.lower() for e in errors)


def test_image_size_case_insensitive():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        for size in ["4k", "4K", "1k", "1K", "2k", "2K", "512"]:
            errors = validate_config(
                subcommand="image",
                prompt="test",
                aspect_ratio=None,
                image_size=size,
                duration_seconds=None,
                output_format="png",
                count=1,
                model="gemini-3.1-flash-image-preview",
                input_image=None,
            )
            assert not errors, f"Unexpected error for size {size}: {errors}"


def test_invalid_image_size():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image",
            prompt="test",
            aspect_ratio=None,
            image_size="8K",
            duration_seconds=None,
            output_format="png",
            count=1,
            model="gemini-3.1-flash-image-preview",
            input_image=None,
        )
    assert any("size" in e.lower() for e in errors)


def test_image_size_not_allowed_for_imagen():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image",
            prompt="test",
            aspect_ratio=None,
            image_size="4K",
            duration_seconds=None,
            output_format="png",
            count=1,
            model="imagen-4.0-generate-001",
            input_image=None,
        )
    assert any("imagen" in e.lower() or "size" in e.lower() for e in errors)


def test_valid_duration():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        for d in [4, 6, 8]:
            errors = validate_config(
                subcommand="video",
                prompt="test",
                aspect_ratio=None,
                image_size=None,
                duration_seconds=d,
                output_format=None,
                count=1,
                model="veo-3.0-generate-001",
                input_image=None,
            )
            assert not errors, f"Unexpected error for duration {d}"


def test_invalid_duration():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="video",
            prompt="test",
            aspect_ratio=None,
            image_size=None,
            duration_seconds=5,
            output_format=None,
            count=1,
            model="veo-3.0-generate-001",
            input_image=None,
        )
    assert any("duration" in e.lower() for e in errors)


def test_invalid_count():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image",
            prompt="test",
            aspect_ratio=None,
            image_size=None,
            duration_seconds=None,
            output_format="png",
            count=0,
            model="gemini-3.1-flash-image-preview",
            input_image=None,
        )
    assert any("count" in e.lower() for e in errors)


def test_unknown_model_is_warning_not_error(capsys):
    """Unknown models produce a warning on stderr but not a validation error."""
    import sys
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image",
            prompt="test",
            aspect_ratio=None,
            image_size=None,
            duration_seconds=None,
            output_format="png",
            count=1,
            model="some-future-model",
            input_image=None,
        )
    assert not errors  # warning, not error


def test_edit_missing_input_image():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="edit",
            prompt="remove background",
            aspect_ratio=None,
            image_size=None,
            duration_seconds=None,
            output_format="png",
            count=1,
            model="gemini-3.1-flash-image-preview",
            input_image="/nonexistent/file.png",
        )
    assert any("input" in e.lower() or "file" in e.lower() for e in errors)


def test_valid_config_no_errors():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        errors = validate_config(
            subcommand="image",
            prompt="a cat",
            aspect_ratio="16:9",
            image_size="4K",
            duration_seconds=None,
            output_format="png",
            count=1,
            model="gemini-3.1-flash-image-preview",
            input_image=None,
        )
    assert errors == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_validation.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write implementation**

`src/genmedia/validation.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_validation.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/genmedia/validation.py tests/unit/test_validation.py
git commit -m "feat: add parameter validation with full rule set"
```

---

### Task 4: Retry Module

**Files:**
- Create: `src/genmedia/retry.py`
- Create: `tests/unit/test_retry.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_retry.py`:
```python
import time
from unittest.mock import MagicMock

import pytest

from genmedia.retry import RetryWrapper, RetryableError, NonRetryableError


def test_succeeds_first_try():
    fn = MagicMock(return_value="result")
    wrapper = RetryWrapper(max_retries=3, base_delay=0.01, max_delay=0.1)
    result = wrapper.execute(fn)
    assert result == "result"
    assert fn.call_count == 1
    assert wrapper.attempts == 1


def test_retries_on_retryable_error():
    fn = MagicMock(side_effect=[RetryableError("429"), RetryableError("429"), "result"])
    wrapper = RetryWrapper(max_retries=5, base_delay=0.01, max_delay=0.1)
    result = wrapper.execute(fn)
    assert result == "result"
    assert fn.call_count == 3
    assert wrapper.attempts == 3


def test_exhausts_retries():
    fn = MagicMock(side_effect=RetryableError("429"))
    wrapper = RetryWrapper(max_retries=3, base_delay=0.01, max_delay=0.1)
    with pytest.raises(RetryableError):
        wrapper.execute(fn)
    assert fn.call_count == 3
    assert wrapper.attempts == 3


def test_does_not_retry_non_retryable():
    fn = MagicMock(side_effect=NonRetryableError("400 bad request"))
    wrapper = RetryWrapper(max_retries=5, base_delay=0.01, max_delay=0.1)
    with pytest.raises(NonRetryableError):
        wrapper.execute(fn)
    assert fn.call_count == 1


def test_respects_retry_after():
    err = RetryableError("429")
    err.retry_after = 0.05
    fn = MagicMock(side_effect=[err, "result"])
    wrapper = RetryWrapper(max_retries=3, base_delay=0.01, max_delay=10.0)
    start = time.monotonic()
    result = wrapper.execute(fn)
    elapsed = time.monotonic() - start
    assert result == "result"
    assert elapsed >= 0.04  # respected retry_after


def test_backoff_increases():
    errors = [RetryableError("429") for _ in range(3)]
    fn = MagicMock(side_effect=errors + ["result"])
    wrapper = RetryWrapper(max_retries=5, base_delay=1.0, max_delay=100.0)
    wrapper.execute(fn)
    # Delays should increase: ~1.0, ~2.0, ~4.0 (with jitter up to 50%)
    # With base 1.0: min delay 1 is 1.0, min delay 2 is 2.0 — always increasing
    assert len(wrapper.delays) == 3
    assert wrapper.delays[1] > wrapper.delays[0]


def test_max_delay_cap():
    wrapper = RetryWrapper(max_retries=10, base_delay=1.0, max_delay=5.0)
    # After several doublings, delay should cap at max_delay
    delay = wrapper._calculate_delay(attempt=10)
    assert delay <= 5.0 * 1.5  # max_delay + jitter


def test_env_var_overrides(monkeypatch):
    monkeypatch.setenv("GENMEDIA_MAX_RETRIES", "2")
    monkeypatch.setenv("GENMEDIA_RETRY_BASE_DELAY", "0.05")
    wrapper = RetryWrapper()
    assert wrapper.max_retries == 2
    assert wrapper.base_delay == 0.05


def test_retryable_error_preserves_status_code():
    err = RetryableError("429 Too Many Requests", status_code=429)
    assert err.status_code == 429


def test_retryable_error_preserves_status_code_500():
    err = RetryableError("500 Internal Server Error", status_code=500)
    assert err.status_code == 500
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_retry.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write implementation**

`src/genmedia/retry.py`:
```python
import os
import random
import time


class RetryableError(Exception):
    def __init__(self, message: str, retry_after: float | None = None, status_code: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.status_code = status_code


class NonRetryableError(Exception):
    pass


def classify_sdk_error(exc: Exception) -> RetryableError | NonRetryableError:
    """Translate a google-genai SDK exception into RetryableError or NonRetryableError."""
    from google.genai import errors as genai_errors

    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    message = str(exc)

    if isinstance(exc, genai_errors.APIError):
        status_code = getattr(exc, "status", status_code)

    retry_after = None
    # Check for Retry-After in response headers if available
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {})
        ra = headers.get("Retry-After") or headers.get("retry-after")
        if ra:
            try:
                retry_after = float(ra)
            except ValueError:
                pass

    if status_code in (429, 500, 503):
        return RetryableError(message, retry_after=retry_after, status_code=status_code)

    return NonRetryableError(message)


class RetryWrapper:
    def __init__(
        self,
        max_retries: int | None = None,
        base_delay: float | None = None,
        max_delay: float = 60.0,
    ):
        self.max_retries = max_retries if max_retries is not None else int(os.environ.get("GENMEDIA_MAX_RETRIES", "5"))
        self.base_delay = base_delay if base_delay is not None else float(os.environ.get("GENMEDIA_RETRY_BASE_DELAY", "2.0"))
        self.max_delay = max_delay
        self.attempts = 0
        self.delays: list[float] = []

    def _calculate_delay(self, attempt: int) -> float:
        delay = self.base_delay * (2 ** (attempt - 1))
        delay = min(delay, self.max_delay)
        jitter = random.uniform(0, delay * 0.5)
        return delay + jitter

    def execute(self, fn):
        self.attempts = 0
        self.delays = []
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            self.attempts = attempt
            try:
                return fn()
            except NonRetryableError:
                raise
            except RetryableError as e:
                last_error = e
                if attempt == self.max_retries:
                    raise
                if e.retry_after is not None:
                    delay = e.retry_after
                else:
                    delay = self._calculate_delay(attempt)
                self.delays.append(delay)
                time.sleep(delay)

        raise last_error
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_retry.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/genmedia/retry.py tests/unit/test_retry.py
git commit -m "feat: add retry wrapper with exponential backoff and jitter"
```

---

### Task 5: Output Module (JSON formatting + auto-naming)

**Files:**
- Create: `src/genmedia/output.py`
- Create: `tests/unit/test_output.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_output.py`:
```python
import json
import os
import tempfile

from genmedia.output import (
    format_success,
    format_error,
    format_dry_run,
    format_list_models,
    auto_name,
    write_media_files,
)


def test_format_success_json():
    result = format_success(
        files=[{"path": "/tmp/genmedia/genmedia_001.png", "mime_type": "image/png", "size_bytes": 1000}],
        model="gemini-3.1-flash-image-preview",
        elapsed_seconds=2.5,
        request={"prompt": "cat", "aspect_ratio": "16:9"},
    )
    parsed = json.loads(result)
    assert parsed["status"] == "success"
    assert len(parsed["files"]) == 1
    assert parsed["model"] == "gemini-3.1-flash-image-preview"
    assert parsed["elapsed_seconds"] == 2.5


def test_format_error_json():
    result = format_error(
        error="rate_limited",
        message="429 after 5 retries",
        retries_attempted=5,
        elapsed_seconds=30.0,
    )
    parsed = json.loads(result)
    assert parsed["status"] == "error"
    assert parsed["error"] == "rate_limited"
    assert parsed["retries_attempted"] == 5


def test_format_error_with_partial_files():
    result = format_error(
        error="file_error",
        message="Disk full",
        retries_attempted=0,
        elapsed_seconds=5.0,
        files=[{"path": "/tmp/genmedia/genmedia_001.png", "mime_type": "image/png", "size_bytes": 1000}],
    )
    parsed = json.loads(result)
    assert parsed["status"] == "error"
    assert len(parsed["files"]) == 1


def test_format_dry_run_json():
    result = format_dry_run(
        backend="GeminiImageBackend",
        sdk_method="client.models.generate_content",
        model="gemini-3.1-flash-image-preview",
        config={"response_modalities": ["IMAGE"]},
        validation_errors=[],
    )
    parsed = json.loads(result)
    assert parsed["status"] == "dry_run"
    assert parsed["backend"] == "GeminiImageBackend"
    assert parsed["validation_errors"] == []


def test_format_dry_run_with_errors():
    result = format_dry_run(
        backend="GeminiImageBackend",
        sdk_method="client.models.generate_content",
        model="gemini-3.1-flash-image-preview",
        config={},
        validation_errors=["bad aspect ratio"],
    )
    parsed = json.loads(result)
    assert parsed["validation_errors"] == ["bad aspect ratio"]


def test_format_list_models_json():
    models = [
        {"id": "model-a", "default": True, "notes": "best"},
        {"id": "model-b", "default": False, "notes": "fast"},
    ]
    result = format_list_models(models)
    parsed = json.loads(result)
    assert len(parsed["models"]) == 2
    assert parsed["models"][0]["id"] == "model-a"


def test_auto_name_no_collision():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = auto_name(output_dir=tmpdir, extension=".png")
        assert path.endswith(".png")
        assert "genmedia_001" in path


def test_auto_name_collision_avoidance():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create existing file
        open(os.path.join(tmpdir, "genmedia_001.png"), "w").close()
        path = auto_name(output_dir=tmpdir, extension=".png")
        assert "genmedia_002" in path


def test_auto_name_creates_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "nested", "dir")
        path = auto_name(output_dir=subdir, extension=".png")
        assert os.path.isdir(subdir)


def test_write_media_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        from genmedia.backends.base import MediaResult
        results = [
            MediaResult(data=b"fake-png-data", mime_type="image/png", metadata={}),
        ]
        written = write_media_files(
            results=results,
            output=None,
            output_dir=tmpdir,
            output_format="png",
        )
        assert len(written) == 1
        assert os.path.isfile(written[0]["path"])
        assert written[0]["size_bytes"] == len(b"fake-png-data")
```

- [ ] **Step 2: Write the base module first (needed by output tests)**

`src/genmedia/backends/base.py`:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MediaResult:
    data: bytes
    mime_type: str
    metadata: dict = field(default_factory=dict)


@dataclass
class MediaConfig:
    prompt: str
    model: str
    aspect_ratio: str | None = None
    image_size: str | None = None
    output_format: str | None = None
    count: int = 1
    duration_seconds: int | None = None
    input_image: bytes | None = None
    input_image_mime: str | None = None


class ContentBlockedError(Exception):
    def __init__(self, message: str, block_reason: str | None = None):
        super().__init__(message)
        self.block_reason = block_reason


class Backend(ABC):
    @abstractmethod
    def build_request(self, config: MediaConfig) -> dict:
        ...

    @abstractmethod
    def validate(self, config: MediaConfig) -> list[str]:
        ...

    @abstractmethod
    def generate(self, config: MediaConfig) -> list[MediaResult]:
        ...
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_output.py -v`
Expected: FAIL (ModuleNotFoundError for `genmedia.output`)

- [ ] **Step 4: Write output module**

`src/genmedia/output.py`:
```python
import json
import os

from genmedia.backends.base import MediaResult

FORMAT_TO_EXT = {"png": ".png", "jpg": ".jpg", "webp": ".webp", "mp4": ".mp4"}
EXT_TO_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp", ".mp4": "video/mp4"}

DEFAULT_OUTPUT_DIR = "/tmp/genmedia"


def format_success(
    *,
    files: list[dict],
    model: str,
    elapsed_seconds: float,
    request: dict,
) -> str:
    return json.dumps(
        {
            "status": "success",
            "files": files,
            "model": model,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "request": request,
        },
        indent=2,
    )


def format_error(
    *,
    error: str,
    message: str,
    retries_attempted: int = 0,
    elapsed_seconds: float = 0.0,
    files: list[dict] | None = None,
    **extra,
) -> str:
    payload = {
        "status": "error",
        "error": error,
        "message": message,
        "retries_attempted": retries_attempted,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }
    if files:
        payload["files"] = files
    payload.update(extra)
    return json.dumps(payload, indent=2)


def format_dry_run(
    *,
    backend: str,
    sdk_method: str,
    model: str,
    config: dict,
    validation_errors: list[str],
) -> str:
    return json.dumps(
        {
            "status": "dry_run",
            "backend": backend,
            "sdk_method": sdk_method,
            "model": model,
            "config": config,
            "validation_errors": validation_errors,
        },
        indent=2,
    )


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


def write_media_files(
    *,
    results: list[MediaResult],
    output: str | None,
    output_dir: str | None,
    output_format: str,
) -> list[dict]:
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

        written.append({
            "path": os.path.abspath(path),
            "mime_type": result.mime_type,
            "size_bytes": len(result.data),
        })

    return written
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_output.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/genmedia/backends/base.py src/genmedia/output.py tests/unit/test_output.py
git commit -m "feat: add output formatting, auto-naming, and base backend classes"
```

---

### Task 6: GeminiImageBackend

**Files:**
- Create: `src/genmedia/backends/gemini.py`
- Create: `tests/unit/test_backends.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_backends.py`:
```python
from unittest.mock import MagicMock, patch

from genmedia.backends.base import MediaConfig, MediaResult, ContentBlockedError
from genmedia.backends.gemini import GeminiImageBackend


class TestGeminiImageBackend:
    def setup_method(self):
        self.client = MagicMock()
        self.backend = GeminiImageBackend(client=self.client)

    def test_build_request_basic(self):
        config = MediaConfig(prompt="a cat", model="gemini-3.1-flash-image-preview")
        req = self.backend.build_request(config)
        assert req["model"] == "gemini-3.1-flash-image-preview"
        assert req["config"]["response_modalities"] == ["IMAGE"]

    def test_build_request_with_aspect_and_size(self):
        config = MediaConfig(
            prompt="a cat",
            model="gemini-3.1-flash-image-preview",
            aspect_ratio="16:9",
            image_size="4K",
        )
        req = self.backend.build_request(config)
        assert req["config"]["image_config"]["aspect_ratio"] == "16:9"
        assert req["config"]["image_config"]["image_size"] == "4K"

    def test_build_request_edit_mode(self):
        config = MediaConfig(
            prompt="remove background",
            model="gemini-3.1-flash-image-preview",
            input_image=b"fake-image-bytes",
            input_image_mime="image/png",
        )
        req = self.backend.build_request(config)
        assert req["config"]["response_modalities"] == ["TEXT", "IMAGE"]
        assert len(req["contents"]) == 2  # text + image part

    def test_generate_success(self):
        # Mock SDK response
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"fake-png-bytes"
        mock_part.inline_data.mime_type = "image/png"
        mock_part.text = None

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.candidates[0].content.parts = [mock_part]

        self.client.models.generate_content.return_value = mock_response

        config = MediaConfig(prompt="a cat", model="gemini-3.1-flash-image-preview")
        results = self.backend.generate(config)
        assert len(results) == 1
        assert results[0].data == b"fake-png-bytes"
        assert results[0].mime_type == "image/png"

    def test_generate_content_blocked_prompt_level(self):
        mock_response = MagicMock()
        mock_response.prompt_feedback = MagicMock()
        mock_response.prompt_feedback.block_reason = "IMAGE_SAFETY"
        mock_response.candidates = []

        self.client.models.generate_content.return_value = mock_response

        config = MediaConfig(prompt="bad prompt", model="gemini-3.1-flash-image-preview")
        with __import__("pytest").raises(ContentBlockedError) as exc_info:
            self.backend.generate(config)
        assert exc_info.value.block_reason == "IMAGE_SAFETY"

    def test_generate_content_blocked_response_level(self):
        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = "SAFETY"
        mock_response.candidates[0].content.parts = []

        self.client.models.generate_content.return_value = mock_response

        config = MediaConfig(prompt="borderline", model="gemini-3.1-flash-image-preview")
        with __import__("pytest").raises(ContentBlockedError):
            self.backend.generate(config)

    def test_generate_count_multiple(self):
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"image-bytes"
        mock_part.inline_data.mime_type = "image/png"
        mock_part.text = None

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.candidates[0].content.parts = [mock_part]

        self.client.models.generate_content.return_value = mock_response

        config = MediaConfig(prompt="a cat", model="gemini-3.1-flash-image-preview", count=3)
        results = self.backend.generate(config)
        assert len(results) == 3
        assert self.client.models.generate_content.call_count == 3

    def test_image_size_normalized_uppercase(self):
        config = MediaConfig(
            prompt="a cat",
            model="gemini-3.1-flash-image-preview",
            image_size="4k",
        )
        req = self.backend.build_request(config)
        assert req["config"]["image_config"]["image_size"] == "4K"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_backends.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write implementation**

`src/genmedia/backends/gemini.py`:
```python
from google.genai import types

from genmedia.backends.base import Backend, ContentBlockedError, MediaConfig, MediaResult
from genmedia.retry import classify_sdk_error


class GeminiImageBackend(Backend):
    def __init__(self, client):
        self.client = client

    def build_request(self, config: MediaConfig) -> dict:
        is_edit = config.input_image is not None

        if is_edit:
            contents = [
                config.prompt,
                types.Part.from_bytes(data=config.input_image, mime_type=config.input_image_mime),
            ]
            modalities = ["TEXT", "IMAGE"]
        else:
            contents = config.prompt
            modalities = ["IMAGE"]

        image_config = {}
        if config.aspect_ratio:
            image_config["aspect_ratio"] = config.aspect_ratio
        if config.image_size:
            normalized = config.image_size.upper() if config.image_size != "512" else "512"
            image_config["image_size"] = normalized

        sdk_config = {"response_modalities": modalities}
        if image_config:
            sdk_config["image_config"] = image_config

        return {
            "model": config.model,
            "contents": contents,
            "config": sdk_config,
        }

    def validate(self, config: MediaConfig) -> list[str]:
        return []

    def generate(self, config: MediaConfig) -> list[MediaResult]:
        results: list[MediaResult] = []

        for _ in range(config.count):
            req = self.build_request(config)

            image_config_obj = None
            if "image_config" in req["config"]:
                image_config_obj = types.ImageConfig(**req["config"]["image_config"])

            try:
                response = self.client.models.generate_content(
                    model=req["model"],
                    contents=req["contents"],
                    config=types.GenerateContentConfig(
                        response_modalities=req["config"]["response_modalities"],
                        image_config=image_config_obj,
                    ),
                )
            except Exception as exc:
                raise classify_sdk_error(exc) from exc

            self._check_safety(response)

            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                    results.append(
                        MediaResult(
                            data=part.inline_data.data,
                            mime_type=getattr(part.inline_data, "mime_type", "image/png"),
                            metadata={},
                        )
                    )

        return results

    def _check_safety(self, response) -> None:
        if response.prompt_feedback and getattr(response.prompt_feedback, "block_reason", None):
            reason = str(response.prompt_feedback.block_reason)
            raise ContentBlockedError(
                f"Prompt blocked by safety filter: {reason}",
                block_reason=reason,
            )

        if response.candidates:
            finish = getattr(response.candidates[0], "finish_reason", None)
            if finish == "SAFETY" or str(finish) == "SAFETY":
                raise ContentBlockedError(
                    "Response blocked by safety filter",
                    block_reason="SAFETY",
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_backends.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/genmedia/backends/gemini.py tests/unit/test_backends.py
git commit -m "feat: add GeminiImageBackend with safety block handling"
```

---

### Task 7: ImagenBackend

**Files:**
- Create: `src/genmedia/backends/imagen.py`
- Modify: `tests/unit/test_backends.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_backends.py`:
```python
from genmedia.backends.imagen import ImagenBackend


class TestImagenBackend:
    def setup_method(self):
        self.client = MagicMock()
        self.backend = ImagenBackend(client=self.client)

    def test_build_request_basic(self):
        config = MediaConfig(prompt="a lake", model="imagen-4.0-generate-001")
        req = self.backend.build_request(config)
        assert req["model"] == "imagen-4.0-generate-001"
        assert req["config"]["number_of_images"] == 1

    def test_build_request_with_count(self):
        config = MediaConfig(prompt="a lake", model="imagen-4.0-generate-001", count=3)
        req = self.backend.build_request(config)
        assert req["config"]["number_of_images"] == 3

    def test_build_request_with_aspect(self):
        config = MediaConfig(prompt="a lake", model="imagen-4.0-generate-001", aspect_ratio="16:9")
        req = self.backend.build_request(config)
        assert req["config"]["aspect_ratio"] == "16:9"

    def test_generate_success(self):
        mock_image = MagicMock()
        mock_image.image.image_bytes = b"imagen-bytes"

        mock_response = MagicMock()
        mock_response.generated_images = [mock_image]

        self.client.models.generate_images.return_value = mock_response

        config = MediaConfig(prompt="a lake", model="imagen-4.0-generate-001")
        results = self.backend.generate(config)
        assert len(results) == 1
        assert results[0].data == b"imagen-bytes"

    def test_generate_uses_native_count(self):
        mock_image = MagicMock()
        mock_image.image.image_bytes = b"imagen-bytes"

        mock_response = MagicMock()
        mock_response.generated_images = [mock_image, mock_image]

        self.client.models.generate_images.return_value = mock_response

        config = MediaConfig(prompt="a lake", model="imagen-4.0-generate-001", count=2)
        results = self.backend.generate(config)
        assert len(results) == 2
        # Only one API call — count handled natively
        assert self.client.models.generate_images.call_count == 1
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/unit/test_backends.py::TestImagenBackend -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write implementation**

`src/genmedia/backends/imagen.py`:
```python
from google.genai import types

from genmedia.backends.base import Backend, MediaConfig, MediaResult
from genmedia.retry import classify_sdk_error


class ImagenBackend(Backend):
    def __init__(self, client):
        self.client = client

    def build_request(self, config: MediaConfig) -> dict:
        imagen_config = {"number_of_images": config.count}

        if config.aspect_ratio:
            imagen_config["aspect_ratio"] = config.aspect_ratio

        if config.output_format:
            mime_map = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}
            imagen_config["output_mime_type"] = mime_map.get(config.output_format, "image/png")

        return {
            "model": config.model,
            "prompt": config.prompt,
            "config": imagen_config,
        }

    def validate(self, config: MediaConfig) -> list[str]:
        errors = []
        if config.image_size:
            errors.append("image_size is not supported with Imagen models")
        return errors

    def generate(self, config: MediaConfig) -> list[MediaResult]:
        req = self.build_request(config)

        try:
            response = self.client.models.generate_images(
                model=req["model"],
                prompt=req["prompt"],
                config=types.GenerateImagesConfig(**req["config"]),
            )
        except Exception as exc:
            raise classify_sdk_error(exc) from exc

        results = []
        for image in response.generated_images:
            results.append(
                MediaResult(
                    data=image.image.image_bytes,
                    mime_type="image/png",
                    metadata={},
                )
            )

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_backends.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/genmedia/backends/imagen.py tests/unit/test_backends.py
git commit -m "feat: add ImagenBackend with native count support"
```

---

### Task 8: VeoBackend

**Files:**
- Create: `src/genmedia/backends/veo.py`
- Modify: `tests/unit/test_backends.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_backends.py`:
```python
from genmedia.backends.veo import VeoBackend


class TestVeoBackend:
    def setup_method(self):
        self.client = MagicMock()
        self.backend = VeoBackend(client=self.client)

    def test_build_request_basic(self):
        config = MediaConfig(prompt="a sunset", model="veo-3.0-generate-001")
        req = self.backend.build_request(config)
        assert req["model"] == "veo-3.0-generate-001"
        assert req["config"]["number_of_videos"] == 1

    def test_build_request_with_duration(self):
        config = MediaConfig(
            prompt="a sunset",
            model="veo-3.0-generate-001",
            duration_seconds=8,
        )
        req = self.backend.build_request(config)
        assert req["config"]["duration_seconds"] == 8

    def test_build_request_with_aspect(self):
        config = MediaConfig(
            prompt="a sunset",
            model="veo-3.0-generate-001",
            aspect_ratio="9:16",
        )
        req = self.backend.build_request(config)
        assert req["config"]["aspect_ratio"] == "9:16"

    def test_generate_success(self):
        # Mock operation that completes immediately
        mock_video = MagicMock()
        mock_video.video = MagicMock()

        mock_operation = MagicMock()
        mock_operation.done = True
        mock_operation.result.generated_videos = [mock_video]

        self.client.models.generate_videos.return_value = mock_operation
        self.client.files.download.return_value = b"video-bytes"

        config = MediaConfig(prompt="a sunset", model="veo-3.0-generate-001")
        results = self.backend.generate(config)
        assert len(results) == 1
        assert results[0].data == b"video-bytes"
        assert results[0].mime_type == "video/mp4"

    def test_generate_polls_until_done(self):
        mock_video = MagicMock()
        mock_video.video = MagicMock()

        # First call: not done. Second call: done.
        pending_op = MagicMock()
        pending_op.done = False

        done_op = MagicMock()
        done_op.done = True
        done_op.result.generated_videos = [mock_video]

        self.client.models.generate_videos.return_value = pending_op
        self.client.operations.get.return_value = done_op
        self.client.files.download.return_value = b"video-bytes"

        config = MediaConfig(prompt="a sunset", model="veo-3.0-generate-001")
        # Patch sleep to avoid waiting
        with patch("genmedia.backends.veo.time.sleep"):
            results = self.backend.generate(config)

        assert len(results) == 1
        self.client.operations.get.assert_called_once()

    def test_generate_keyboard_interrupt(self):
        pending_op = MagicMock()
        pending_op.done = False

        self.client.models.generate_videos.return_value = pending_op
        self.client.operations.get.side_effect = KeyboardInterrupt()

        config = MediaConfig(prompt="a sunset", model="veo-3.0-generate-001")
        with patch("genmedia.backends.veo.time.sleep"):
            with __import__("pytest").raises(KeyboardInterrupt):
                self.backend.generate(config)
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/unit/test_backends.py::TestVeoBackend -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write implementation**

`src/genmedia/backends/veo.py`:
```python
import time

from google.genai import types

from genmedia.backends.base import Backend, MediaConfig, MediaResult
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

        return {
            "model": config.model,
            "prompt": config.prompt,
            "config": veo_config,
        }

    def validate(self, config: MediaConfig) -> list[str]:
        return []

    def generate(self, config: MediaConfig) -> list[MediaResult]:
        req = self.build_request(config)

        try:
            operation = self.client.models.generate_videos(
                model=req["model"],
                prompt=req["prompt"],
                config=types.GenerateVideosConfig(**req["config"]),
            )
        except Exception as exc:
            raise classify_sdk_error(exc) from exc

        operation = self._poll_operation(operation)

        results = []
        for video in operation.result.generated_videos:
            video_bytes = self.client.files.download(file=video.video)
            results.append(
                MediaResult(
                    data=video_bytes,
                    mime_type="video/mp4",
                    metadata={},
                )
            )

        return results

    def _poll_operation(self, operation):
        while not operation.done:
            time.sleep(self.POLL_INTERVAL)
            operation = self.client.operations.get(operation)
        return operation
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_backends.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/genmedia/backends/veo.py tests/unit/test_backends.py
git commit -m "feat: add VeoBackend with polling loop"
```

---

### Task 9: CLI Main Group + Image Subcommand

**Files:**
- Create: `src/genmedia/cli/main.py`
- Create: `src/genmedia/cli/image.py`
- Create: `tests/unit/test_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_cli.py`:
```python
import json
import os
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from genmedia.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner(mix_stderr=False)


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


class TestImageCommand:
    def test_missing_api_key(self, runner):
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(cli, ["image", "a cat"])
        assert result.exit_code == 2
        err = json.loads(result.stderr)
        assert err["error"] == "validation_error"
        assert "GEMINI_API_KEY" in err["message"]

    def test_empty_prompt(self, runner, mock_env):
        result = runner.invoke(cli, ["image", ""])
        assert result.exit_code == 2

    def test_dry_run(self, runner, mock_env):
        result = runner.invoke(cli, ["image", "a cat", "--dry-run"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "dry_run"
        assert parsed["backend"] == "GeminiImageBackend"
        assert parsed["model"] == "gemini-3.1-flash-image-preview"

    def test_dry_run_imagen(self, runner, mock_env):
        result = runner.invoke(cli, ["image", "a cat", "--model", "imagen-4.0-generate-001", "--dry-run"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["backend"] == "ImagenBackend"

    def test_dry_run_with_validation_errors(self, runner, mock_env):
        result = runner.invoke(cli, ["image", "a cat", "--aspect", "99:1", "--dry-run"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed["validation_errors"]) > 0

    def test_list_models(self, runner):
        # Does not require API key
        result = runner.invoke(cli, ["image", "--list-models"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "models" in parsed
        assert any(m["default"] for m in parsed["models"])

    def test_invalid_aspect_ratio(self, runner, mock_env):
        result = runner.invoke(cli, ["image", "a cat", "--aspect", "7:3"])
        assert result.exit_code == 2
        err = json.loads(result.stderr)
        assert err["error"] == "validation_error"

    @patch("genmedia.cli.image.genai")
    def test_successful_generation(self, mock_genai, runner, mock_env, tmp_path):
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"fake-png"
        mock_part.inline_data.mime_type = "image/png"
        mock_part.text = None

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.candidates[0].content.parts = [mock_part]

        mock_genai.Client.return_value.models.generate_content.return_value = mock_response

        result = runner.invoke(cli, [
            "image", "a cat",
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"
        assert len(parsed["files"]) == 1
        assert os.path.isfile(parsed["files"][0]["path"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_cli.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write CLI main group**

`src/genmedia/cli/main.py`:
```python
import click


@click.group()
@click.version_option(package_name="genmedia")
def cli():
    """GenMedia — multimodal media generation CLI for Google GenAI."""
    pass


# Import subcommands to register them
from genmedia.cli.image import image  # noqa: E402
from genmedia.cli.edit import edit  # noqa: E402
from genmedia.cli.video import video  # noqa: E402

cli.add_command(image)
cli.add_command(edit)
cli.add_command(video)
```

- [ ] **Step 4: Write image subcommand**

`src/genmedia/cli/image.py`:
```python
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
```

- [ ] **Step 5: Create stub edit and video commands** (so imports don't fail)

`src/genmedia/cli/edit.py`:
```python
import click


@click.command()
def edit():
    """Edit/inpaint an existing image."""
    click.echo("Not yet implemented")
```

`src/genmedia/cli/video.py`:
```python
import click


@click.command()
def video():
    """Generate video using Veo models."""
    click.echo("Not yet implemented")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_cli.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/genmedia/cli/ tests/unit/test_cli.py
git commit -m "feat: add CLI main group and image subcommand with full pipeline"
```

---

### Task 10: Edit Subcommand

**Files:**
- Modify: `src/genmedia/cli/edit.py`
- Modify: `tests/unit/test_cli.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_cli.py`:
```python
class TestEditCommand:
    def test_dry_run_edit(self, runner, mock_env, tmp_path):
        # Create a fake input image
        img_path = tmp_path / "input.png"
        img_path.write_bytes(b"fake-png-data")

        result = runner.invoke(cli, ["edit", str(img_path), "remove background", "--dry-run"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "dry_run"
        assert parsed["backend"] == "GeminiImageBackend"
        assert parsed["config"]["response_modalities"] == ["TEXT", "IMAGE"]

    def test_edit_missing_input_file(self, runner, mock_env):
        result = runner.invoke(cli, ["edit", "/nonexistent.png", "remove bg"])
        assert result.exit_code == 2

    def test_edit_missing_prompt(self, runner, mock_env, tmp_path):
        img_path = tmp_path / "input.png"
        img_path.write_bytes(b"fake-png")
        result = runner.invoke(cli, ["edit", str(img_path)])
        assert result.exit_code != 0

    @patch("genmedia.cli.edit.genai")
    def test_edit_success(self, mock_genai, runner, mock_env, tmp_path):
        img_path = tmp_path / "input.png"
        img_path.write_bytes(b"fake-input")

        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"edited-png"
        mock_part.inline_data.mime_type = "image/png"
        mock_part.text = None

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.candidates[0].content.parts = [mock_part]

        mock_genai.Client.return_value.models.generate_content.return_value = mock_response

        result = runner.invoke(cli, [
            "edit", str(img_path), "remove background",
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_cli.py::TestEditCommand -v`
Expected: FAIL

- [ ] **Step 3: Implement edit subcommand**

`src/genmedia/cli/edit.py`:
```python
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
@click.argument("input_image")
@click.argument("prompt")
@click.option("--model", "-m", default=None, help="Model ID")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--output-dir", "-d", default=None, help="Output directory")
@click.option("--count", "-n", default=1, type=int, help="Number of variations")
@click.option("--aspect", "-a", default=None, help="Override aspect ratio")
@click.option("--size", "-s", default=None, help="Override image size")
@click.option("--format", "-f", "output_format", default="png", help="Output format: png, jpg, webp")
@click.option("--verbose", "-v", is_flag=True, help="Extra metadata")
@click.option("--pretty", is_flag=True, help="Human-friendly output")
@click.option("--dry-run", is_flag=True, help="Show request without calling API")
def edit(input_image, prompt, model, output, output_dir, count, aspect, size, output_format, verbose, pretty, dry_run):
    """Edit/inpaint an existing image."""
    model = model or get_default_model("edit")

    errors = validate_config(
        subcommand="edit",
        prompt=prompt,
        aspect_ratio=aspect,
        image_size=size,
        duration_seconds=None,
        output_format=output_format,
        count=count,
        model=model,
        input_image=input_image,
    )

    if dry_run:
        image_bytes = b""
        mime = "image/png"
        if os.path.isfile(input_image):
            image_bytes = open(input_image, "rb").read()
            mime = mimetypes.guess_type(input_image)[0] or "image/png"

        backend = GeminiImageBackend(client=None)
        config = MediaConfig(
            prompt=prompt,
            model=model,
            aspect_ratio=aspect,
            image_size=size,
            output_format=output_format,
            count=count,
            input_image=image_bytes,
            input_image_mime=mime,
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
        _exit_error("validation_error", "; ".join(errors), exit_code=2)

    image_bytes = open(input_image, "rb").read()
    mime = mimetypes.guess_type(input_image)[0] or "image/png"

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    backend = GeminiImageBackend(client=client)

    config = MediaConfig(
        prompt=prompt,
        model=model,
        aspect_ratio=aspect,
        image_size=size,
        output_format=output_format,
        count=count,
        input_image=image_bytes,
        input_image_mime=mime,
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

    request_info = {"prompt": prompt, "input_image": input_image}
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_cli.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/genmedia/cli/edit.py tests/unit/test_cli.py
git commit -m "feat: add edit subcommand (image editing via Gemini)"
```

---

### Task 11: Video Subcommand

**Files:**
- Modify: `src/genmedia/cli/video.py`
- Modify: `tests/unit/test_cli.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_cli.py`:
```python
class TestVideoCommand:
    def test_dry_run_video(self, runner, mock_env):
        result = runner.invoke(cli, ["video", "a sunset", "--dry-run"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "dry_run"
        assert parsed["backend"] == "VeoBackend"
        assert parsed["model"] == "veo-3.0-generate-001"

    def test_dry_run_with_duration(self, runner, mock_env):
        result = runner.invoke(cli, ["video", "a sunset", "--duration", "8", "--dry-run"])
        parsed = json.loads(result.output)
        assert parsed["config"]["duration_seconds"] == 8

    def test_invalid_duration(self, runner, mock_env):
        result = runner.invoke(cli, ["video", "a sunset", "--duration", "5"])
        assert result.exit_code == 2
        err = json.loads(result.stderr)
        assert err["error"] == "validation_error"

    def test_list_video_models(self, runner):
        result = runner.invoke(cli, ["video", "--list-models"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert any(m["id"].startswith("veo") for m in parsed["models"])

    @patch("genmedia.cli.video.genai")
    def test_video_success(self, mock_genai, runner, mock_env, tmp_path):
        mock_video = MagicMock()
        mock_video.video = MagicMock()

        mock_operation = MagicMock()
        mock_operation.done = True
        mock_operation.result.generated_videos = [mock_video]

        mock_genai.Client.return_value.models.generate_videos.return_value = mock_operation
        mock_genai.Client.return_value.files.download.return_value = b"fake-video"

        result = runner.invoke(cli, [
            "video", "a sunset",
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"
        assert parsed["files"][0]["mime_type"] == "video/mp4"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_cli.py::TestVideoCommand -v`
Expected: FAIL

- [ ] **Step 3: Implement video subcommand**

`src/genmedia/cli/video.py`:
```python
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
@click.option("--verbose", "-v", is_flag=True, help="Extra metadata")
@click.option("--pretty", is_flag=True, help="Human-friendly output")
@click.option("--dry-run", is_flag=True, help="Show request without calling API")
@click.option("--list-models", is_flag=True, help="List available video models")
def video(prompt, model, output, output_dir, count, aspect, duration, verbose, pretty, dry_run, list_models):
    """Generate video using Veo models."""
    if list_models:
        click.echo(format_list_models(VIDEO_MODELS))
        sys.exit(0)

    if prompt is None:
        _exit_error("validation_error", "Prompt is required (use --list-models to list models without a prompt)", exit_code=2)

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
        _exit_error("validation_error", "; ".join(errors), exit_code=2)

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
        _exit_error("cancelled", "Polling cancelled. The server-side operation may still be running.", elapsed_seconds=elapsed, exit_code=1)
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
            output_format="mp4",
        )
    except OSError as e:
        _exit_error("file_error", str(e), exit_code=3)

    request_info = {"prompt": prompt}
    if aspect:
        request_info["aspect_ratio"] = aspect
    if duration:
        request_info["duration_seconds"] = duration

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_cli.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/genmedia/cli/video.py tests/unit/test_cli.py
git commit -m "feat: add video subcommand with Veo polling and Ctrl+C handling"
```

---

### Task 12: Integration Test Scaffolding

**Files:**
- Create: `tests/integration/conftest.py`
- Create: `tests/integration/test_gemini_image.py`
- Create: `tests/integration/test_imagen.py`
- Create: `tests/integration/test_veo.py`

- [ ] **Step 1: Create conftest gate**

`tests/integration/conftest.py`:
```python
import os
import pytest


def pytest_collection_modifyitems(config, items):
    if not os.environ.get("GENMEDIA_TEST_LIVE"):
        skip = pytest.mark.skip(reason="Set GENMEDIA_TEST_LIVE=1 to run integration tests")
        for item in items:
            if "integration" in str(item.fspath):
                item.add_marker(skip)
```

- [ ] **Step 2: Create integration test stubs**

`tests/integration/test_gemini_image.py`:
```python
import json
import os
import tempfile

from click.testing import CliRunner

from genmedia.cli.main import cli


def test_gemini_image_happy_path():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(cli, [
            "image", "a simple red circle on white background",
            "--output-dir", tmpdir,
            "--size", "512",
        ])
        assert result.exit_code == 0, f"Failed: {result.output}"
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"
        assert len(parsed["files"]) == 1
        assert os.path.isfile(parsed["files"][0]["path"])
        assert parsed["files"][0]["size_bytes"] > 0
```

`tests/integration/test_imagen.py`:
```python
import json
import os
import tempfile

from click.testing import CliRunner

from genmedia.cli.main import cli


def test_imagen_happy_path():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(cli, [
            "image", "a simple blue square",
            "--model", "imagen-4.0-generate-001",
            "--output-dir", tmpdir,
        ])
        assert result.exit_code == 0, f"Failed: {result.output}"
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"
        assert len(parsed["files"]) == 1
```

`tests/integration/test_veo.py`:
```python
import json
import os
import tempfile

from click.testing import CliRunner

from genmedia.cli.main import cli


def test_veo_happy_path():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(cli, [
            "video", "a slow pan across a calm lake at dawn",
            "--duration", "4",
            "--output-dir", tmpdir,
        ])
        assert result.exit_code == 0, f"Failed: {result.output}"
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"
        assert len(parsed["files"]) == 1
        assert parsed["files"][0]["mime_type"] == "video/mp4"
```

- [ ] **Step 3: Verify integration tests are skipped without env var**

Run: `pytest tests/integration/ -v`
Expected: All SKIPPED with "Set GENMEDIA_TEST_LIVE=1" message

- [ ] **Step 4: Commit**

```bash
git add tests/integration/
git commit -m "feat: add integration test scaffolding with live gate"
```

---

### Task 13: End-to-End Smoke Test

**Files:**
- None new — uses existing CLI

- [ ] **Step 1: Run full unit test suite**

Run: `pytest tests/unit/ -v`
Expected: All PASS

- [ ] **Step 2: Test CLI entry point**

Run: `genmedia --version`
Expected: Prints version

Run: `genmedia --help`
Expected: Shows subcommands: image, edit, video

Run: `genmedia image --help`
Expected: Shows image flags

- [ ] **Step 3: Test dry-run end-to-end**

Run: `GEMINI_API_KEY=fake genmedia image "test" --dry-run`
Expected: JSON with `status: dry_run`

Run: `GEMINI_API_KEY=fake genmedia video "test" --dry-run --duration 8`
Expected: JSON with `status: dry_run`, backend VeoBackend

Run: `genmedia image --list-models`
Expected: JSON with model list (no API key needed)

- [ ] **Step 4: Test validation errors**

Run: `genmedia image "test"`  (no API key set)
Expected: Exit 2, JSON error about GEMINI_API_KEY

Run: `GEMINI_API_KEY=fake genmedia image "test" --aspect 99:1`
Expected: Exit 2, JSON error about aspect ratio

- [ ] **Step 5: Commit any fixes if needed, then final commit**

```bash
git add -A
git commit -m "chore: fix any issues found during smoke testing"
```

(Only if fixes were needed. Skip if everything passed clean.)

---

### Task 14: Additional Test Coverage

**Files:**
- Modify: `tests/unit/test_cli.py` (append tests)
- Modify: `tests/integration/` (add edit integration test)

- [ ] **Step 1: Write additional CLI tests**

Append to `tests/unit/test_cli.py`:
```python
class TestAdditionalCoverage:
    def test_explicit_output_path(self, runner, mock_env, tmp_path):
        """Test --output flag writes to exact path."""
        with patch("genmedia.cli.image.genai") as mock_genai:
            mock_part = MagicMock()
            mock_part.inline_data = MagicMock()
            mock_part.inline_data.data = b"fake-png"
            mock_part.inline_data.mime_type = "image/png"
            mock_part.text = None

            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_response.candidates[0].finish_reason = "STOP"
            mock_response.candidates[0].content.parts = [mock_part]

            mock_genai.Client.return_value.models.generate_content.return_value = mock_response

            out_path = str(tmp_path / "exact.png")
            result = runner.invoke(cli, ["image", "a cat", "--output", out_path])
            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert parsed["files"][0]["path"] == out_path
            assert os.path.isfile(out_path)

    def test_count_multiple_files(self, runner, mock_env, tmp_path):
        """Test --count 3 produces 3 files."""
        with patch("genmedia.cli.image.genai") as mock_genai:
            mock_part = MagicMock()
            mock_part.inline_data = MagicMock()
            mock_part.inline_data.data = b"fake-png"
            mock_part.inline_data.mime_type = "image/png"
            mock_part.text = None

            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_response.candidates[0].finish_reason = "STOP"
            mock_response.candidates[0].content.parts = [mock_part]

            mock_genai.Client.return_value.models.generate_content.return_value = mock_response

            result = runner.invoke(cli, [
                "image", "a cat", "--count", "3", "--output-dir", str(tmp_path),
            ])
            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert len(parsed["files"]) == 3
            for f in parsed["files"]:
                assert os.path.isfile(f["path"])

    def test_format_jpg(self, runner, mock_env, tmp_path):
        """Test --format jpg produces .jpg extension."""
        with patch("genmedia.cli.image.genai") as mock_genai:
            mock_part = MagicMock()
            mock_part.inline_data = MagicMock()
            mock_part.inline_data.data = b"fake-jpg"
            mock_part.inline_data.mime_type = "image/jpeg"
            mock_part.text = None

            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_response.candidates[0].finish_reason = "STOP"
            mock_response.candidates[0].content.parts = [mock_part]

            mock_genai.Client.return_value.models.generate_content.return_value = mock_response

            result = runner.invoke(cli, [
                "image", "a cat", "--format", "jpg", "--output-dir", str(tmp_path),
            ])
            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert parsed["files"][0]["path"].endswith(".jpg")
```

- [ ] **Step 2: Run new tests**

Run: `pytest tests/unit/test_cli.py::TestAdditionalCoverage -v`
Expected: All PASS

- [ ] **Step 3: Add edit integration test**

`tests/integration/test_edit.py`:
```python
import json
import os
import tempfile

from click.testing import CliRunner

from genmedia.cli.main import cli


def test_edit_happy_path(tmp_path):
    """Create a simple image first, then edit it."""
    runner = CliRunner(mix_stderr=False)

    # First generate an image to use as input
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(cli, [
            "image", "a solid red square on white background",
            "--output-dir", tmpdir,
            "--size", "512",
        ])
        assert result.exit_code == 0, f"Image gen failed: {result.output}"
        gen_result = json.loads(result.output)
        input_path = gen_result["files"][0]["path"]

        # Now edit it
        result = runner.invoke(cli, [
            "edit", input_path, "make the square blue instead of red",
            "--output-dir", tmpdir,
        ])
        assert result.exit_code == 0, f"Edit failed: {result.output}"
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"
        assert len(parsed["files"]) == 1
        assert os.path.isfile(parsed["files"][0]["path"])
```

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: add coverage for --output, --count, --format, and edit integration"
```

---

### Task 15: Pretty + Verbose Mode (deferred scope)

> **Note:** `--pretty` and `--verbose` flags are accepted by the CLI but produce no special behavior in v1. This task adds basic implementations. Can be expanded later.

**Files:**
- Modify: `src/genmedia/output.py` (add pretty formatting functions)
- Modify: `src/genmedia/cli/image.py`, `edit.py`, `video.py` (wire pretty flag)
- Create: `tests/unit/test_pretty.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/test_pretty.py`:
```python
from genmedia.output import format_pretty_success, format_pretty_error


def test_pretty_success_no_json():
    result = format_pretty_success(
        files=[{"path": "/tmp/genmedia/genmedia_001.png", "mime_type": "image/png", "size_bytes": 1000}],
        model="gemini-3.1-flash-image-preview",
        elapsed_seconds=2.5,
    )
    assert "Saved to" in result
    assert "/tmp/genmedia/genmedia_001.png" in result
    assert "{" not in result  # no JSON


def test_pretty_error_no_json():
    result = format_pretty_error(
        error="rate_limited",
        message="429 after retries",
    )
    assert "Error" in result or "error" in result
    assert "429" in result
    assert "{" not in result  # no JSON


def test_pretty_success_multiple_files():
    files = [
        {"path": "/tmp/genmedia/genmedia_001.png", "mime_type": "image/png", "size_bytes": 1000},
        {"path": "/tmp/genmedia/genmedia_002.png", "mime_type": "image/png", "size_bytes": 1200},
    ]
    result = format_pretty_success(files=files, model="test", elapsed_seconds=3.0)
    assert "genmedia_001" in result
    assert "genmedia_002" in result
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_pretty.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Add pretty formatting functions to output.py**

Append to `src/genmedia/output.py`:
```python
def format_pretty_success(
    *,
    files: list[dict],
    model: str,
    elapsed_seconds: float,
) -> str:
    lines = []
    for f in files:
        size_kb = f["size_bytes"] / 1024
        lines.append(f"Saved to {f['path']} ({size_kb:.1f} KB)")
    lines.append(f"Model: {model} | Time: {elapsed_seconds:.1f}s")
    return "\n".join(lines)


def format_pretty_error(*, error: str, message: str) -> str:
    return f"Error [{error}]: {message}"
```

- [ ] **Step 4: Wire pretty flag in image subcommand**

In `src/genmedia/cli/image.py`, after the success path where `format_success` is called, add:

```python
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
```

Apply the same pattern to `edit.py` and `video.py`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/genmedia/output.py src/genmedia/cli/ tests/unit/test_pretty.py
git commit -m "feat: add basic --pretty mode for human-friendly output"
```
