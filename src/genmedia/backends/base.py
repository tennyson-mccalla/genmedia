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
