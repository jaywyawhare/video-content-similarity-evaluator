from pydantic import BaseModel, AnyUrl, validator, constr
from typing import List, Dict, Optional


class S3URL(AnyUrl):
    allowed_schemes = {"s3"}   


class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None
    error_code: Optional[str] = None


class VideoRatingRequest(BaseModel):
    master_url: S3URL
    candidate_urls: List[S3URL]

    @validator("candidate_urls")
    def validate_urls(cls, v, values):
        if not v:
            raise ValueError("At least one candidate URL is required")
        if "master_url" in values and values["master_url"] in v:
            raise ValueError("Master URL cannot be in candidate URLs")
        return v


class VideoScore(BaseModel):
    url: S3URL
    keyword_coverage: float
    content_similarity: float
    keyword_similarity: float
    semantic_similarity: float
    sentence_structure: float
    readability_score: float
    filler_word_penalty: float
    aggregate_score: float
    error: Optional[str] = None
    error_code: Optional[str] = None


class RatingResponse(BaseModel):
    results: List[VideoScore]


class VideoUploadResponse(BaseModel):
    url: str
    size: Optional[int] = None
    content_type: Optional[str] = None
    duration: Optional[float] = None
    request_id: Optional[str] = None

    @validator("url")
    def validate_s3_url(cls, v):
        if not v.startswith("s3://"):
            raise ValueError("URL must be an S3 URL")
        return v


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class ReadinessResponse(BaseModel):
    status: str
    checks: Dict[str, str]
