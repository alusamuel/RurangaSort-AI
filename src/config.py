"""Central configuration for RurangaSort AI, loaded from environment variables / .env."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RurangaSort AI"
    environment: str = "development"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501
    api_base_url: str = "http://localhost:8000"

    api_key: str = "change-me-local-dev-key"

    model_dir: str = "models"
    model_path: str = "models/active/model.h5"
    class_names_path: str = "models/class_names.json"

    image_size: int = 224
    num_channels: int = 3

    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "sqlite:///./rurangasort.db"

    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    data_train_dir: str = "data/train"
    data_validation_dir: str = "data/validation"
    data_test_dir: str = "data/test"
    upload_directory: str = "data/uploaded"

    max_upload_size_mb: int = 200
    max_zip_files: int = 5000

    promotion_min_macro_f1_delta: float = -0.01
    promotion_max_latency_ms: float = 500.0
    promotion_max_per_class_recall_drop: float = 0.05

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = ""
    aws_s3_bucket: str = ""

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return (self.image_size, self.image_size, self.num_channels)

    def resolve(self, relative_path: str) -> Path:
        """Resolve a path from the settings relative to the project root."""
        path = Path(relative_path)
        return path if path.is_absolute() else ROOT_DIR / path


settings = Settings()

DEFAULT_CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/bmp"}
