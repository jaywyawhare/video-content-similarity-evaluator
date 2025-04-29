import os
import boto3
import tempfile
import time
from typing import Dict, Any
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from botocore.config import Config  
from fastapi import UploadFile, HTTPException
from mimetypes import guess_type
from tenacity import retry, stop_after_attempt, wait_exponential
from ..config import settings
from ..core.logger import get_logger

logger = get_logger(__name__)


class S3Service:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(S3Service, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            session = boto3.Session(
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_DEFAULT_REGION,
            )
            self._client = session.client(
                "s3",
                endpoint_url=settings.AWS_ENDPOINT,
                config=Config(
                    request_checksum_calculation="when_required",
                    response_checksum_validation="when_required",
                ),
            )
        self.s3_client = self._client

    def check_connection(self) -> bool:
        """Check if S3 connection is working"""
        try:
            self.s3_client.head_bucket(Bucket=settings.AWS_BUCKET)
            return True
        except ClientError as e:
            raise Exception(f"Failed to connect to S3: {str(e)}")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def get_file_metadata(self, url: str) -> Dict[str, Any]:
        """Get file metadata from S3"""
        try:
            bucket, key = self._parse_s3_url(url)
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            return {
                "content_type": response.get("ContentType"),
                "content_length": response.get("ContentLength", 0),
                "last_modified": response.get("LastModified"),
            }
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                raise FileNotFoundError(f"File not found in S3: {url}")
            elif error_code == "NoSuchBucket":
                raise ValueError(f"Bucket not found: {bucket}")
            raise Exception(f"Failed to get file metadata from S3: {str(e)}")

    async def validate_video_file(self, url: str) -> bool:
        """Validate if file is a video and within size limits"""
        try:
            _, ext = os.path.splitext(url)
            ext = ext.lower()
            if not ext:
                raise ValueError("Missing file extension")
            if ext not in settings.ALLOWED_FILE_EXTENSIONS:
                raise ValueError(
                    f"Invalid file extension: {ext}. Allowed: {', '.join(settings.ALLOWED_FILE_EXTENSIONS)}"
                )

            metadata = self.get_file_metadata(url)
            content_type = metadata.get("content_type")

            if not content_type:
                content_type = guess_type(url)[0]
                if not content_type:
                    ext_to_type = {
                        ".mp4": "video/mp4",
                        ".mpeg": "video/mpeg",
                        ".mov": "video/quicktime",
                        ".avi": "video/x-msvideo",
                        ".wmv": "video/x-ms-wmv",
                    }
                    content_type = ext_to_type.get(ext)

            if not content_type:
                raise ValueError("Could not determine content type")

            if content_type not in settings.ALLOWED_VIDEO_TYPES:
                raise ValueError(f"Invalid file type: {content_type}")

            content_length = metadata.get("content_length", 0)
            if not content_length:
                raise ValueError("Could not determine file size")

            if content_length > settings.MAX_UPLOAD_SIZE:
                raise ValueError(
                    f"File too large: {content_length} bytes (max {settings.MAX_UPLOAD_SIZE})"
                )

            return True

        except ClientError as e:
            raise ValueError(f"S3 validation failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"File validation failed: {str(e)}")

    def download_file(self, url: str, local_path: str) -> None:
        """Download file from S3 to local path"""
        try:
            bucket, key = self._parse_s3_url(url)
            self.s3_client.download_file(bucket, key, local_path)

            # Verify download was successful
            if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                raise Exception("Download failed: File is empty or missing")

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                raise FileNotFoundError(f"File not found in S3: {url}")
            elif error_code == "NoSuchBucket":
                raise ValueError(f"Bucket not found")
            raise Exception(f"Failed to download file: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to download file: {str(e)}")

    def upload_to_bucket(self, file_path: str, object_name: str) -> str:
        """Direct S3 upload using put_object"""
        try:
            if not os.path.isfile(file_path):
                raise Exception(f"File not found: {file_path}")

            # Add videos/ prefix to object name
            s3_key = f"videos/{object_name.lstrip('/')}"

            with open(file_path, "rb") as f:
                self.s3_client.put_object(
                    Bucket=settings.AWS_BUCKET, Key=s3_key, Body=f
                )
            return f"s3://{settings.AWS_BUCKET}/{s3_key}"

        except NoCredentialsError:
            raise Exception("AWS credentials not found")
        except PartialCredentialsError:
            raise Exception("Incomplete AWS credentials")
        except Exception as e:
            raise Exception(f"Failed to upload file: {str(e)}")

    async def upload_video_to_s3(self, video: UploadFile, s3_key: str) -> str:
        """Handle FastAPI video upload with enhanced validation and logging"""
        temp_path = None
        start_time = time.time()

        try:
            # Validate file size before reading
            content_length = video.size if hasattr(video, "size") else 0
            if content_length > settings.MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE} bytes",
                )

            suffix = os.path.splitext(video.filename)[1]
            with tempfile.NamedTemporaryFile(
                delete=False, prefix="upload_", suffix=suffix
            ) as tmp:
                temp_path = tmp.name
                size = 0

                # Read in chunks with progress logging
                while chunk := await video.read(1024 * 1024):  # 1MB chunks
                    size += len(chunk)
                    if size > settings.MAX_UPLOAD_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE} bytes",
                        )
                    tmp.write(chunk)

                tmp.flush()
                os.fsync(tmp.fileno())

            # Validate temp file
            if not os.path.isfile(temp_path):
                raise HTTPException(
                    status_code=500, detail="Failed to create temporary file"
                )

            if os.path.getsize(temp_path) == 0:
                raise HTTPException(status_code=400, detail="Empty file uploaded")

            # Upload to S3
            s3_url = self.upload_to_bucket(temp_path, s3_key)

            duration = time.time() - start_time
            logger.info(
                "S3 upload successful",
                extra={
                    "duration": duration,
                    "file_size": size,
                    "s3_url": s3_url,
                },
            )

            return s3_url

        except HTTPException:
            raise

        except Exception as e:
            logger.error("S3 upload failed", extra={"error": str(e)}, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.error(f"Failed to cleanup temp file: {str(e)}")

    @staticmethod
    def _parse_s3_url(url: str) -> tuple[str, str]:
        path = url.replace("s3://", "")
        bucket = path.split("/")[0]
        key = "/".join(path.split("/")[1:])
        return bucket, key

    @classmethod
    def cleanup(cls):
        """Clean up S3 client resources"""
        if cls._instance and cls._instance._client:
            try:
                cls._instance._client.close()
                cls._instance._client = None
            except Exception as e:
                logger.error(f"Error cleaning up S3 client: {str(e)}")
