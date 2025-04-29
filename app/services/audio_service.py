import os
import asyncio
import subprocess
import time
from typing import Optional
from ..core.logger import get_logger

logger = get_logger(__name__)


class AudioService:
    async def convert_video_to_audio(
        self, video_path: str, output_path: Optional[str] = None
    ) -> str:
        """Convert video to audio using ffmpeg"""
        if output_path is None:
            output_path = f"{os.path.splitext(video_path)[0]}.wav"

        try:
            # Validate input file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Run ffmpeg asynchronously
            try:
                await asyncio.to_thread(self._run_ffmpeg, video_path, output_path)
            except Exception as e:
                # Clean up partial output file if it exists
                if os.path.exists(output_path):
                    os.remove(output_path)
                raise e

            # Validate output was created
            if not os.path.exists(output_path):
                raise RuntimeError("Failed to create output audio file")

            return output_path

        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg error converting video to audio: {error_message}")
            raise RuntimeError(f"Failed to convert video to audio: {error_message}")
        except Exception as e:
            logger.error(f"Error in convert_video_to_audio: {str(e)}")
            raise

    def _run_ffmpeg(self, video_path: str, output_path: str) -> None:
        """Run ffmpeg command using subprocess"""
        try:
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Basic ffmpeg command with simpler options
            command = [
                "ffmpeg",
                "-i",
                video_path,
                "-vn",  # No video
                "-acodec",
                "pcm_s16le",  # PCM 16-bit encoding
                "-ar",
                "16000",  # 16kHz sample rate
                "-ac",
                "1",  # Mono audio
                "-y",  # Overwrite output
                output_path,
            ]

            logger.info(
                "Running ffmpeg command",
                extra={
                    "input_path": video_path,
                    "output_path": output_path,
                    "command": " ".join(command),
                },
            )

            # Run ffmpeg with output capture
            result = subprocess.run(command, capture_output=True, text=True, check=True)

            if os.path.getsize(output_path) == 0:
                raise RuntimeError("Generated audio file is empty")

            logger.info(
                "Audio extraction successful",
                extra={
                    "input_path": video_path,
                    "output_path": output_path,
                    "output_size": os.path.getsize(output_path),
                },
            )

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr
            logger.error(f"FFmpeg error: {error_msg}")
            raise RuntimeError(f"FFmpeg processing failed: {error_msg}")
        except Exception as e:
            logger.error("Error in ffmpeg processing", exc_info=True)
            raise RuntimeError(f"Error processing video: {str(e)}")

    def validate_audio_file(self, audio_path: str) -> bool:
        """Validate audio file format and quality"""
        try:
            probe = ffmpeg.probe(audio_path)
            audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]

            if not audio_streams:
                raise ValueError("No audio streams found")

            audio_info = audio_streams[0]
            duration = float(probe["format"]["duration"])

            # Validate audio duration and format
            if duration < 0.1:
                raise ValueError("Audio file too short")

            if audio_info.get("sample_rate", "0") == "0":
                raise ValueError("Invalid audio sample rate")

            return True

        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            raise ValueError(f"Invalid audio file: {error_message}")
