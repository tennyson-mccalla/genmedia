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
