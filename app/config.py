import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    api_key: str
    match_threshold: float = 0.40
    model_name: str = "buffalo_l"
    data_dir: Path = Path("data")
    min_face_size: int = 80


def load_settings() -> Settings:
    api_key = os.environ.get("API_KEY", "")
    if not api_key:
        raise RuntimeError("API_KEY environment variable is required")
    return Settings(
        api_key=api_key,
        match_threshold=float(os.environ.get("MATCH_THRESHOLD", "0.40")),
        model_name=os.environ.get("MODEL_NAME", "buffalo_l"),
        data_dir=Path(os.environ.get("DATA_DIR", "data")),
        min_face_size=int(os.environ.get("MIN_FACE_SIZE", "80")),
    )
