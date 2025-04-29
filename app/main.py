from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
from botocore.exceptions import ClientError
from .core.logger import get_logger
from .core.logging_config import setup_logging
from .config import settings
from .models import (
    VideoRatingRequest,
    RatingResponse,
    HealthResponse,
    ReadinessResponse,
    VideoUploadResponse,
)
from .services.audio_service import AudioService
from .services.transcription_service import TranscriptionService
from .services.rating_service import RatingService
from .services.s3_service import S3Service  # Update import to only S3Service
import uuid
import tempfile
import os
import subprocess
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
import time

# Initialize logging
setup_logging()
logger = get_logger(__name__)  # updated logger usage

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Video Rating Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add metrics
Instrumentator().instrument(app).expose(app)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()

    # Log request details
    logger.info(
        f"Incoming {request.method} request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "endpoint": str(request.url.path),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        },
    )

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        # Log response details
        logger.info(
            f"Request completed with status {response.status_code}",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration": duration,
            },
        )

        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        logger.error(
            "Request failed",
            extra={
                "request_id": request_id,
                "error": str(e),
                "duration": time.time() - start_time,
            },
            exc_info=True,
        )
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    try:
        # Check S3 connection
        s3_service = S3Service()
        s3_service.check_connection()

        # Initialize Whisper model and warm it up
        transcription_service = TranscriptionService()
        await transcription_service.load_model()
        await transcription_service.warmup()

        # Initialize spaCy model
        rating_service = RatingService()
        if rating_service.nlp is None:
            raise RuntimeError("Failed to load spaCy model")

        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    try:
        # Cleanup S3 connections
        S3Service.cleanup()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for basic liveness probe"""
    return HealthResponse(status="healthy", timestamp=datetime.utcnow().isoformat())


@app.get("/readiness", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    """Readiness check endpoint to verify all dependencies are available"""
    try:
        # Check S3 connection
        s3_service = S3Service()
        s3_service.check_connection()

        # Check if ffmpeg is available
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)

        # Check if whisper model can be loaded
        transcription_service = TranscriptionService()
        await transcription_service.load_model()

        return ReadinessResponse(
            status="ready",
            checks={"s3": "ok", "ffmpeg": "ok", "whisper": "ok"},
        )
    except subprocess.SubprocessError:
        return ReadinessResponse(
            status="not_ready",
            checks={"s3": "ok", "ffmpeg": "error", "whisper": "unknown"},
        )
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return ReadinessResponse(
            status="not_ready",
            checks={
                "s3": "error" if isinstance(e, ClientError) else "unknown",
                "ffmpeg": "unknown",
                "whisper": "error" if isinstance(e, RuntimeError) else "unknown",
            },
        )


@app.post("/upload_video/", response_model=VideoUploadResponse)
async def upload_video(video: UploadFile = File(...), request: Request = None):
    start_time = time.time()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.info(
        "Starting video upload",
        extra={
            "request_id": request_id,
            "upload_name": video.filename,  # Changed from filename to upload_name
            "content_type": video.content_type,
            "endpoint": "/upload_video",
        },
    )

    try:
        # Validate content type
        if video.content_type not in settings.ALLOWED_VIDEO_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type. Allowed types: {settings.ALLOWED_VIDEO_TYPES}",
            )

        # Generate secure random filename with original extension
        ext = os.path.splitext(video.filename)[1].lower()
        if ext not in settings.ALLOWED_FILE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension. Allowed extensions: {settings.ALLOWED_FILE_EXTENSIONS}",
            )

        unique_filename = f"{uuid.uuid4()}{ext}"
        s3_service = S3Service()
        s3_url = await s3_service.upload_video_to_s3(video, unique_filename)

        duration = time.time() - start_time
        logger.info(
            "Video upload successful",
            extra={
                "request_id": request_id,
                "duration": duration,
                "s3_url": s3_url,
                "upload_name": video.filename,  # Changed from upload_filename to upload_name
            },
        )

        return VideoUploadResponse(
            url=s3_url,
            size=video.size if hasattr(video, "size") else None,
            content_type=video.content_type,
            duration=duration,
            request_id=request_id,
        )

    except HTTPException as e:
        logger.error(
            "Video upload failed with HTTP error",
            extra={
                "request_id": request_id,
                "status_code": e.status_code,
                "detail": e.detail,
                "upload_name": video.filename,  # Changed from upload_filename to upload_name
            },
        )
        raise

    except Exception as e:
        logger.error(
            "Video upload failed with unexpected error",
            extra={
                "request_id": request_id,
                "error": str(e),
                "upload_name": video.filename,  # Changed from upload_filename to upload_name
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error during upload",
        )


@app.post("/rate_videos/", response_model=RatingResponse)
async def rate_videos(request: VideoRatingRequest, req: Request = None):
    request_id = getattr(req.state, "request_id", str(uuid.uuid4()))

    with logger.perf_track("rate_videos"):
        logger.info(
            "Starting video rating",
            extra={
                "request_id": request_id,
                "master_url": str(request.master_url),
                "candidate_count": len(request.candidate_urls),
                "endpoint": "/rate_videos",
            },
        )
        files_to_cleanup = []
        try:
            # Validate URLs
            if not all(
                str(url).startswith("s3://")
                for url in [request.master_url] + request.candidate_urls
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid S3 URL format",
                    headers={"X-Error-Code": "INVALID_URL_FORMAT"},
                )

            s3_service = S3Service()
            # Initialize results list with VideoScore objects for each candidate
            results = [
                {
                    "url": str(url),
                    "keyword_coverage": 0.0,
                    "content_similarity": 0.0,
                    "filler_word_penalty": 0.0,
                    "semantic_similarity": 0.0,
                    "aggregate_score": 0.0,
                    "error": None,
                }
                for url in request.candidate_urls
            ]

            with tempfile.TemporaryDirectory() as temp_dir:
                # Process master video
                master_video_path = os.path.join(temp_dir, "master.mp4")
                try:
                    s3_service.download_file(str(request.master_url), master_video_path)
                    master_audio_path = await AudioService().convert_video_to_audio(
                        master_video_path
                    )
                    files_to_cleanup.extend([master_video_path, master_audio_path])
                    master_transcript = await TranscriptionService().transcribe(
                        master_audio_path
                    )

                    for idx, candidate_url in enumerate(request.candidate_urls):
                        try:
                            candidate_video_path = os.path.join(
                                temp_dir, f"candidate_{idx}.mp4"
                            )
                            files_to_cleanup.append(candidate_video_path)
                            s3_service.download_file(
                                str(candidate_url), candidate_video_path
                            )
                            candidate_audio_path = (
                                await AudioService().convert_video_to_audio(
                                    candidate_video_path
                                )
                            )
                            files_to_cleanup.append(candidate_audio_path)
                            candidate_transcript = (
                                await TranscriptionService().transcribe(
                                    candidate_audio_path
                                )
                            )

                            scores = RatingService().calculate_scores(
                                master_transcript, candidate_transcript
                            )
                            results[idx].update(scores)
                        except Exception as e:
                            logger.error(
                                f"Error processing candidate {candidate_url}: {str(e)}",
                                exc_info=True,
                            )
                            results[idx].update(
                                {"error": str(e), "error_code": "PROCESSING_ERROR"}
                            )
                            continue

                except Exception as e:
                    logger.error(
                        f"Error processing master video: {str(e)}", exc_info=True
                    )
                    raise HTTPException(status_code=500, detail=str(e))

            return RatingResponse(results=results)

        except Exception as e:
            logger.error(
                "Error processing video rating request",
                extra={"request_id": req.state.request_id, "error": str(e)},
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            for file_path in files_to_cleanup:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error cleaning up file {file_path}: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
