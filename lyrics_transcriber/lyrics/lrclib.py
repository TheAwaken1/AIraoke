"""
LRCLIB Lyrics Provider - Fetches lyrics from lrclib.net API

LRCLIB provides both synced (timestamped) and plain lyrics for free
without requiring an API key.
"""

import logging
import re
import requests
from typing import Optional, Dict, Any, List
from lyrics_transcriber.types import LyricsData, LyricsMetadata, LyricsSegment, Word
from lyrics_transcriber.lyrics.base_lyrics_provider import BaseLyricsProvider, LyricsProviderConfig
from lyrics_transcriber.utils.word_utils import WordUtils


class LRCLIBProvider(BaseLyricsProvider):
    """Fetches lyrics from LRCLIB (lrclib.net) - free, no API key required."""

    API_BASE = "https://lrclib.net/api"

    def __init__(self, config: LyricsProviderConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AIraoke/1.0 (https://github.com/karaoke-transcriber)"
        })

    def _fetch_data_from_source(self, artist: str, title: str) -> Optional[Dict[str, Any]]:
        """Fetch lyrics from LRCLIB API."""
        self.logger.info(f"Searching LRCLIB for {artist} - {title}")

        # Try direct get first (faster if exact match)
        direct_result = self._try_direct_get(artist, title)
        if direct_result:
            return direct_result

        # Fall back to search
        search_result = self._try_search(artist, title)
        if search_result:
            return search_result

        self.logger.warning(f"No lyrics found on LRCLIB for {artist} - {title}")
        return None

    def _try_direct_get(self, artist: str, title: str) -> Optional[Dict[str, Any]]:
        """Try to get lyrics directly by artist and title."""
        try:
            url = f"{self.API_BASE}/get"
            params = {
                "artist_name": artist,
                "track_name": title
            }

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get("syncedLyrics") or data.get("plainLyrics"):
                    self.logger.info("Found lyrics via direct LRCLIB lookup")
                    return data

            return None

        except Exception as e:
            self.logger.debug(f"Direct LRCLIB lookup failed: {e}")
            return None

    def _try_search(self, artist: str, title: str) -> Optional[Dict[str, Any]]:
        """Search for lyrics if direct lookup fails."""
        try:
            url = f"{self.API_BASE}/search"
            params = {"q": f"{artist} {title}"}

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                results = response.json()
                if results and len(results) > 0:
                    # Find best match (prefer synced lyrics)
                    best_match = None
                    for result in results:
                        if result.get("syncedLyrics"):
                            best_match = result
                            break
                        elif result.get("plainLyrics") and not best_match:
                            best_match = result

                    if best_match:
                        self.logger.info("Found lyrics via LRCLIB search")
                        return best_match

            return None

        except Exception as e:
            self.logger.error(f"LRCLIB search failed: {e}")
            return None

    def _convert_result_format(self, raw_data: Dict[str, Any]) -> LyricsData:
        """Convert LRCLIB response to standardized LyricsData format."""

        # Check if we have synced lyrics (with timestamps)
        synced_lyrics = raw_data.get("syncedLyrics")
        plain_lyrics = raw_data.get("plainLyrics", "")

        is_synced = bool(synced_lyrics)

        # Create metadata
        metadata = LyricsMetadata(
            source="lrclib",
            track_name=raw_data.get("trackName", raw_data.get("name", "")),
            artist_names=raw_data.get("artistName", ""),
            album_name=raw_data.get("albumName"),
            duration_ms=int(raw_data.get("duration", 0) * 1000) if raw_data.get("duration") else None,
            is_synced=is_synced,
            lyrics_provider="lrclib",
            lyrics_provider_id=str(raw_data.get("id", "")),
            provider_metadata={
                "instrumental": raw_data.get("instrumental", False),
                "lrclib_id": raw_data.get("id"),
            }
        )

        # Parse lyrics into segments
        if is_synced:
            segments = self._parse_synced_lyrics(synced_lyrics)
        else:
            segments = self._create_segments_with_words(plain_lyrics, is_synced=False)

        return LyricsData(
            source="lrclib",
            segments=segments,
            metadata=metadata
        )

    def _parse_synced_lyrics(self, synced_lyrics: str) -> List[LyricsSegment]:
        """Parse LRC format synced lyrics into segments with word timing."""
        segments = []
        lines = synced_lyrics.strip().split("\n")

        # LRC format: [MM:SS.ms] lyrics text
        lrc_pattern = re.compile(r'\[(\d{2}):(\d{2})\.(\d{2,3})\]\s*(.*)')

        parsed_lines = []
        for line in lines:
            match = lrc_pattern.match(line.strip())
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                # Handle both 2-digit and 3-digit milliseconds
                ms_str = match.group(3)
                if len(ms_str) == 2:
                    milliseconds = int(ms_str) * 10
                else:
                    milliseconds = int(ms_str)

                timestamp = minutes * 60 + seconds + milliseconds / 1000.0
                text = match.group(4).strip()

                if text:  # Skip empty lines
                    parsed_lines.append((timestamp, text))

        # Create segments with estimated word timing
        for i, (start_time, text) in enumerate(parsed_lines):
            # Estimate end time from next line or add 3 seconds
            if i + 1 < len(parsed_lines):
                end_time = parsed_lines[i + 1][0]
            else:
                end_time = start_time + 3.0

            # Create words with distributed timing
            words = self._create_words_with_timing(text, start_time, end_time)

            segment = LyricsSegment(
                id=WordUtils.generate_id(),
                text=text,
                words=words,
                start_time=start_time,
                end_time=end_time
            )
            segments.append(segment)

        return segments

    def _create_words_with_timing(self, text: str, start_time: float, end_time: float) -> List[Word]:
        """Create Word objects with proportionally distributed timing."""
        word_texts = text.split()
        if not word_texts:
            return []

        duration = end_time - start_time
        total_chars = sum(len(w) for w in word_texts)
        if total_chars == 0:
            total_chars = len(word_texts)

        words = []
        current_time = start_time

        for word_text in word_texts:
            # Distribute time proportionally by character count
            word_duration = (len(word_text) / total_chars) * duration
            word_duration = max(word_duration, 0.05)  # Minimum 50ms per word
            word_end = min(current_time + word_duration, end_time)

            words.append(Word(
                id=WordUtils.generate_id(),
                text=word_text,
                start_time=current_time,
                end_time=word_end,
                confidence=1.0,  # Reference lyrics are ground truth
                created_during_correction=False
            ))
            current_time = word_end

        # Ensure last word ends at segment end
        if words:
            words[-1] = Word(
                id=words[-1].id,
                text=words[-1].text,
                start_time=words[-1].start_time,
                end_time=end_time,
                confidence=1.0,
                created_during_correction=False
            )

        return words
