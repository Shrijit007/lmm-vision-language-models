from dataclasses import dataclass
from pathlib import Path
import mimetypes
import time

from PIL import Image


@dataclass
class ModelResult:
    provider: str
    model: str
    answer: str
    elapsed_seconds: float


def validate_image(image_path: str) -> tuple[Path, str]:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/jpeg", "image/png"}:
        raise ValueError("Use a PNG, JPG or JPEG image.")

    # Verify that the file is a readable image.
    with Image.open(path) as image:
        image.verify()

    return path, mime_type


def timed_call(function):
    start = time.perf_counter()
    answer = function()
    elapsed = time.perf_counter() - start
    return answer, elapsed
