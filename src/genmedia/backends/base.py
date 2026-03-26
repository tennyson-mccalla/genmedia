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
    last_frame_image: bytes | None = None
    last_frame_mime: str | None = None
    resolution: str | None = None
    enhance_prompt: bool = False
    style_ref: bytes | None = None
    style_ref_mime: str | None = None
    asset_refs: list[tuple[bytes, str]] | None = None  # list of (bytes, mime_type)


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
