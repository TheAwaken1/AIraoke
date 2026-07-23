"""
Automatic Lyrics Alignment - Aligns fetched lyrics to audio using Whisper timestamps.

This module combines:
1. Whisper transcription (accurate timing from audio)
2. Fetched lyrics (accurate text from LRCLIB/Genius)

Result: Perfect sync with correct lyrics text.
"""

import logging
import re
from typing import List, Tuple, Optional
from difflib import SequenceMatcher
from lyrics_transcriber.types import LyricsSegment, Word
from lyrics_transcriber.utils.word_utils import WordUtils

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove punctuation)."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text


def similarity_ratio(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def find_best_match(target_words: List[str], whisper_words: List[Tuple[str, float, float]],
                    start_idx: int = 0) -> Tuple[int, int, float]:
    """
    Find the best matching position in whisper_words for a sequence of target_words.

    Returns: (start_index, end_index, confidence)
    """
    if not target_words or not whisper_words:
        return (start_idx, start_idx, 0.0)

    target_text = " ".join(target_words)
    target_normalized = normalize_text(target_text)

    best_score = 0.0
    best_start = start_idx
    best_end = start_idx

    # Sliding window search
    window_size = len(target_words)
    search_range = min(len(whisper_words) - start_idx, window_size * 3 + 10)

    for i in range(start_idx, min(start_idx + search_range, len(whisper_words))):
        # Try different window sizes around expected length
        for size_offset in range(-2, 3):
            end = min(i + window_size + size_offset, len(whisper_words))
            if end <= i:
                continue

            whisper_text = " ".join(w[0] for w in whisper_words[i:end])
            score = similarity_ratio(target_text, whisper_text)

            if score > best_score:
                best_score = score
                best_start = i
                best_end = end

    return (best_start, best_end, best_score)


def align_word_level(fetched_words: List[str], whisper_words: List[Tuple[str, float, float]],
                     segment_start: float, segment_end: float) -> List[Word]:
    """
    Align individual words from fetched lyrics to whisper timestamps.

    Args:
        fetched_words: List of words from fetched lyrics
        whisper_words: List of (word, start_time, end_time) from Whisper
        segment_start: Segment start time
        segment_end: Segment end time

    Returns:
        List of Word objects with aligned timing
    """
    if not fetched_words:
        return []

    if not whisper_words:
        # No whisper words - distribute timing evenly
        duration = segment_end - segment_start
        word_duration = duration / len(fetched_words)

        words = []
        current_time = segment_start
        for word_text in fetched_words:
            words.append(Word(
                id=WordUtils.generate_id(),
                text=word_text,
                start_time=current_time,
                end_time=current_time + word_duration,
                confidence=0.5
            ))
            current_time += word_duration
        return words

    # Use dynamic programming to align words
    aligned_words = []
    whisper_idx = 0

    for i, fetched_word in enumerate(fetched_words):
        fetched_normalized = normalize_text(fetched_word)

        # Find best matching whisper word
        best_match_idx = whisper_idx
        best_score = 0.0

        # Search in a window around current position
        search_end = min(whisper_idx + 5, len(whisper_words))
        for j in range(whisper_idx, search_end):
            whisper_normalized = normalize_text(whisper_words[j][0])
            score = similarity_ratio(fetched_normalized, whisper_normalized)

            if score > best_score:
                best_score = score
                best_match_idx = j

        if best_score > 0.5 and best_match_idx < len(whisper_words):
            # Good match found - use whisper timing
            w = whisper_words[best_match_idx]
            aligned_words.append(Word(
                id=WordUtils.generate_id(),
                text=fetched_word,  # Use fetched text (more accurate)
                start_time=w[1],
                end_time=w[2],
                confidence=best_score
            ))
            whisper_idx = best_match_idx + 1
        else:
            # No good match - interpolate timing
            if aligned_words:
                prev_end = aligned_words[-1].end_time
            else:
                prev_end = segment_start

            # Estimate timing based on position
            remaining_words = len(fetched_words) - i
            remaining_time = segment_end - prev_end
            word_duration = remaining_time / remaining_words if remaining_words > 0 else 0.1

            aligned_words.append(Word(
                id=WordUtils.generate_id(),
                text=fetched_word,
                start_time=prev_end,
                end_time=prev_end + word_duration,
                confidence=0.3
            ))

    return aligned_words


def align_lyrics_to_audio(
    fetched_segments: List[LyricsSegment],
    whisper_segments: List[LyricsSegment],
    whisper_words: Optional[List[Word]] = None
) -> List[LyricsSegment]:
    """
    Align fetched lyrics to audio using Whisper transcription for timing.

    This combines:
    - Fetched lyrics: Accurate text (from LRCLIB, Genius, etc.)
    - Whisper output: Accurate timing (from audio analysis)

    Args:
        fetched_segments: Segments from lyrics provider (accurate text)
        whisper_segments: Segments from Whisper (accurate timing)
        whisper_words: Optional word-level timing from Whisper

    Returns:
        Aligned segments with fetched text and Whisper timing
    """
    if not fetched_segments:
        logger.warning("No fetched segments to align")
        return whisper_segments

    if not whisper_segments:
        logger.warning("No Whisper segments for alignment - using fetched timing")
        return fetched_segments

    logger.info(f"Aligning {len(fetched_segments)} fetched segments to {len(whisper_segments)} Whisper segments")

    # Build list of all whisper words with timing
    all_whisper_words = []
    for seg in whisper_segments:
        for word in seg.words:
            all_whisper_words.append((word.text, word.start_time, word.end_time))

    if not all_whisper_words:
        # Fall back to segment-level alignment
        logger.warning("No word-level Whisper data - using segment-level alignment")
        return align_segments_only(fetched_segments, whisper_segments)

    logger.info(f"Total Whisper words for alignment: {len(all_whisper_words)}")

    # Align each fetched segment to whisper words
    aligned_segments = []
    whisper_word_idx = 0

    for fetched_seg in fetched_segments:
        fetched_words = fetched_seg.text.split()

        if not fetched_words:
            continue

        # Find best matching position in whisper words
        start_idx, end_idx, confidence = find_best_match(
            fetched_words,
            all_whisper_words,
            whisper_word_idx
        )

        if confidence > 0.3:
            # Good match - use whisper timing
            segment_start = all_whisper_words[start_idx][1]
            segment_end = all_whisper_words[min(end_idx - 1, len(all_whisper_words) - 1)][2]

            # Align words within segment
            segment_whisper_words = all_whisper_words[start_idx:end_idx]
            aligned_words = align_word_level(
                fetched_words,
                segment_whisper_words,
                segment_start,
                segment_end
            )

            aligned_segments.append(LyricsSegment(
                id=fetched_seg.id,
                text=fetched_seg.text,
                words=aligned_words,
                start_time=segment_start,
                end_time=segment_end
            ))

            # Move search position forward
            whisper_word_idx = end_idx

            logger.debug(f"Aligned: '{fetched_seg.text[:30]}...' -> {segment_start:.2f}s-{segment_end:.2f}s (conf={confidence:.2f})")
        else:
            # Poor match - try to interpolate
            logger.warning(f"Low confidence alignment for: '{fetched_seg.text[:30]}...' (conf={confidence:.2f})")

            # Use fetched timing as fallback but try to anchor to nearby whisper
            if aligned_segments:
                # Start after previous segment
                estimated_start = aligned_segments[-1].end_time + 0.1
            elif whisper_word_idx < len(all_whisper_words):
                # Use next whisper word as anchor
                estimated_start = all_whisper_words[whisper_word_idx][1]
            else:
                estimated_start = fetched_seg.start_time

            # Estimate duration based on word count
            estimated_duration = len(fetched_words) * 0.3  # ~300ms per word
            estimated_end = estimated_start + estimated_duration

            aligned_words = align_word_level(
                fetched_words,
                [],  # No whisper words to match
                estimated_start,
                estimated_end
            )

            aligned_segments.append(LyricsSegment(
                id=fetched_seg.id,
                text=fetched_seg.text,
                words=aligned_words,
                start_time=estimated_start,
                end_time=estimated_end
            ))

    logger.info(f"Alignment complete: {len(aligned_segments)} segments aligned")
    return aligned_segments


def align_segments_only(
    fetched_segments: List[LyricsSegment],
    whisper_segments: List[LyricsSegment]
) -> List[LyricsSegment]:
    """
    Fallback: Align at segment level when word-level data isn't available.
    """
    aligned = []
    whisper_idx = 0

    for fetched in fetched_segments:
        best_match = None
        best_score = 0.0

        # Search for best matching whisper segment
        for i in range(whisper_idx, min(whisper_idx + 10, len(whisper_segments))):
            score = similarity_ratio(fetched.text, whisper_segments[i].text)
            if score > best_score:
                best_score = score
                best_match = whisper_segments[i]
                whisper_idx = i + 1

        if best_match and best_score > 0.4:
            # Use whisper timing with fetched text
            words = []
            word_texts = fetched.text.split()
            duration = best_match.end_time - best_match.start_time
            word_duration = duration / len(word_texts) if word_texts else duration

            current_time = best_match.start_time
            for word_text in word_texts:
                words.append(Word(
                    id=WordUtils.generate_id(),
                    text=word_text,
                    start_time=current_time,
                    end_time=current_time + word_duration,
                    confidence=best_score
                ))
                current_time += word_duration

            aligned.append(LyricsSegment(
                id=fetched.id,
                text=fetched.text,
                words=words,
                start_time=best_match.start_time,
                end_time=best_match.end_time
            ))
        else:
            # Keep original fetched timing
            aligned.append(fetched)

    return aligned
