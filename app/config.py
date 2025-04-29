from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Set


class Settings(BaseSettings):
    # AWS Settings
    AWS_ACCESS_KEY_ID: str = Field(..., env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    AWS_DEFAULT_REGION: str = Field(..., env="AWS_DEFAULT_REGION")
    AWS_BUCKET: str = Field(..., env="AWS_BUCKET")
    AWS_ENDPOINT: str = Field(..., env="AWS_ENDPOINT")
    AWS_USE_PATH_STYLE_ENDPOINT: bool = Field(
        default=True, env="AWS_USE_PATH_STYLE_ENDPOINT"
    )
    AWS_MAX_RETRIES: int = Field(default=3, env="AWS_MAX_RETRIES")
    AWS_CONNECT_TIMEOUT: int = Field(default=5, env="AWS_CONNECT_TIMEOUT")
    AWS_READ_TIMEOUT: int = Field(default=300, env="AWS_READ_TIMEOUT")

    # API Settings
    API_RATE_LIMIT: int = 100  # requests per minute
    MAX_CONCURRENT_PROCESSES: int = 3
    REQUEST_TIMEOUT: int = 300  # seconds

    # File Settings
    ALLOWED_VIDEO_TYPES: Set[str] = {
        "video/mp4",
        "video/mpeg",
        "video/quicktime",
        "video/x-msvideo",
        "video/x-ms-wmv",
    }
    MAX_UPLOAD_SIZE: int = 500_000_000  # 500MB in bytes
    ALLOWED_FILE_EXTENSIONS: Set[str] = {".mp4", ".mpeg", ".mov", ".avi", ".wmv"}

    # Whisper Settings
    WHISPER_MODEL: str = "tiny.en"

    @field_validator(
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_BUCKET", mode="after"
    )
    def validate_required_fields(cls, v, info):
        if not v:
            raise ValueError(f"{info.field_name} cannot be empty")
        return v

    @field_validator("AWS_BUCKET")
    def validate_bucket_name(cls, v):
        if not v:
            raise ValueError("S3 bucket name cannot be empty")
        if len(v) < 3 or len(v) > 63:
            raise ValueError("S3 bucket name must be between 3 and 63 characters")
        return v

    @field_validator("MAX_UPLOAD_SIZE", mode="after")
    def validate_max_upload_size(cls, v, info):
        if v <= 0:
            raise ValueError("MAX_UPLOAD_SIZE must be positive")
        return v

    @field_validator("API_RATE_LIMIT", mode="after")
    def validate_rate_limit(cls, v, info):
        if v <= 0:
            raise ValueError("API_RATE_LIMIT must be positive")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )


settings = Settings()
