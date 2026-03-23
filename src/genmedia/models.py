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
