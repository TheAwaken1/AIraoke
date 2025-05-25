#! /usr/bin/env python3
from dataclasses import dataclass
import os
import json
import hashlib
import tempfile
import time
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from pydub import AudioSegment
import whisper
import logging
from .base_transcriber import BaseTranscriber, TranscriptionError
import warnings
from lyrics_transcriber.types import TranscriptionData, LyricsSegment, Word
from lyrics_transcriber.utils.word_utils import WordUtils

# Suppress FutureWarning from torch.load inside whisper.load_model
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

@dataclass
class WhisperConfig:
    """Configuration for Whisper transcription service."""
    language: str = "en"
    task: str = "transcribe"
    model_size: str = "turbo"
    VALID_MODELS: tuple = ("turbo", "large-v3")

class WhisperTranscriber(BaseTranscriber):
    """Transcription service using local Whisper model."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        config: Optional[WhisperConfig] = None,
        logger: Optional[Any] = None,
    ):
        """Initialize Whisper transcriber."""
        super().__init__(cache_dir=cache_dir, logger=logger)
        self.config = config or WhisperConfig()
        self.logger = logger or logging.getLogger(__name__)

        if self.config.model_size not in self.config.VALID_MODELS:
            self.logger.error(f"Invalid model size: {self.config.model_size}. Valid options: {self.config.VALID_MODELS}")
            raise ValueError(f"Invalid model size: {self.config.model_size}")

        try:
            self.logger.info(f"Loading Whisper model: {self.config.model_size}")
            self.model = whisper.load_model(self.config.model_size)
            self.logger.info(f"Successfully loaded Whisper model: {self.config.model_size}")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model {self.config.model_size}: {str(e)}")
            raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

    def get_name(self) -> str:
        return "Whisper"

    def _perform_transcription(self, audio_filepath: str) -> Dict[str, Any]:
        """Perform transcription using local Whisper model."""
        self.logger.info(f"Starting transcription for {audio_filepath}")
        try:
            result = self.model.transcribe(
                audio_filepath,
                language=self.config.language,
                task=self.config.task,
                word_timestamps=True
            )
            return result
        except Exception as e:
            self.logger.error(f"Transcription error: {str(e)}")
            raise TranscriptionError(f"Transcription error: {str(e)}")

    def _aggregate_words_into_segments(self, words: List[Word], text: str, max_duration: float = 60.0, min_segment_duration: float = 2.0) -> List[LyricsSegment]:
        """Aggregate word-level data into segments if segments are not provided."""
        if not words:
            self.logger.warning("No words provided to aggregate into segments.")
            return []

        segments = []
        current_segment_words = []
        current_text = []
        start_time = words[0].start_time
        last_end_time = start_time

        for word in words:
            # Check if the gap between words is too large, or if the segment is too long
            if (word.start_time - last_end_time > 2.0) or (word.start_time - start_time > max_duration):
                if current_segment_words:
                    # Close the current segment
                    segment_text = " ".join(current_text).strip()
                    segments.append(
                        LyricsSegment(
                            id=WordUtils.generate_id(),
                            text=segment_text,
                            words=current_segment_words,
                            start_time=start_time,
                            end_time=last_end_time,
                        )
                    )
                # Start a new segment
                current_segment_words = [word]
                current_text = [word.text]
                start_time = word.start_time
            else:
                current_segment_words.append(word)
                current_text.append(word.text)
            last_end_time = word.end_time

        # Close the last segment
        if current_segment_words:
            segment_text = " ".join(current_text).strip()
            segments.append(
                LyricsSegment(
                    id=WordUtils.generate_id(),
                    text=segment_text,
                    words=current_segment_words,
                    start_time=start_time,
                    end_time=last_end_time,
                )
            )

        # If no segments were created or the text doesn't match, create a fallback segment
        if not segments and text.strip():
            segments.append(
                LyricsSegment(
                    id=WordUtils.generate_id(),
                    text=text.strip(),
                    words=words,
                    start_time=0.0,
                    end_time=max_duration,
                )
            )

        # Filter out segments that are too short
        segments = [
            seg for seg in segments
            if (seg.end_time - seg.start_time) >= min_segment_duration
        ]

        self.logger.debug(f"Aggregated {len(segments)} segments from word-level data")
        return segments

    def _convert_result_format(self, raw_data: Dict[str, Any]) -> TranscriptionData:
        """Convert Whisper response to standard format."""
        self._validate_response(raw_data)

        all_words = []

        # Collect all words from word_timestamps
        word_list = []
        for segment in raw_data.get("segments", []):
            for word in segment.get("words", []):
                word_list.append(
                    Word(
                        id=WordUtils.generate_id(),
                        text=word["word"].strip(),
                        start_time=word["start"],
                        end_time=word["end"],
                        confidence=word.get("probability"),
                    )
                )
        all_words.extend(word_list)

        # Create segments using words within each segment's time range
        segments = []
        for seg in raw_data["segments"]:
            segment_words = [word for word in word_list if seg["start"] <= word.start_time < seg["end"]]
            segments.append(
                LyricsSegment(
                    id=WordUtils.generate_id(),
                    text=seg["text"].strip(),
                    words=segment_words,
                    start_time=seg["start"],
                    end_time=seg["end"],
                )
            )

        # If no segments were created but words are present, aggregate words into segments
        if not segments and all_words:
            self.logger.warning("No segments found in raw Whisper output. Aggregating words into segments.")
            segments = self._aggregate_words_into_segments(all_words, raw_data["text"])

        # If still no segments, create a fallback segment from the text
        if not segments and raw_data["text"].strip():
            self.logger.warning("No segments available. Creating a fallback segment from the text output.")
            segments.append(
                LyricsSegment(
                    id=WordUtils.generate_id(),
                    text=raw_data["text"].strip(),
                    words=all_words,
                    start_time=0.0,
                    end_time=60.0,
                )
            )

        return TranscriptionData(
            segments=segments,
            words=all_words,
            text=raw_data["text"],
            source=self.get_name(),
            metadata={
                "language": raw_data.get("language", self.config.language),
                "model": self.config.model_size,
            },
        )

    def _validate_response(self, raw_data: Dict[str, Any]) -> None:
        """Validate the response contains required fields."""
        if "text" not in raw_data:
            raise TranscriptionError("Response missing required 'text' field")