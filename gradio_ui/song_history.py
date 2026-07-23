"""
Song History Manager - Tracks recently processed songs
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SongHistoryManager:
    """Manages song history tracking for AIraoke."""

    MAX_HISTORY_SIZE = 50  # Maximum number of songs to keep in history

    def __init__(self, history_file: Optional[str] = None):
        """Initialize the song history manager.

        Args:
            history_file: Path to the history JSON file. Defaults to app/gradio_ui/song_history.json
        """
        if history_file is None:
            history_file = os.path.join(os.path.dirname(__file__), "song_history.json")

        self.history_file = history_file
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create the history file if it doesn't exist."""
        if not os.path.exists(self.history_file):
            self._save_history([])

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load history from file."""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error loading history file: {e}. Starting fresh.")
            return []

    def _save_history(self, history: List[Dict[str, Any]]):
        """Save history to file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving history file: {e}")

    def add_song(
        self,
        artist: str,
        title: str,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
        lyrics_source: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add a song to the history.

        Args:
            artist: Artist name
            title: Song title
            audio_path: Path to the audio file
            video_path: Path to the generated video
            lyrics_source: Source of lyrics (lrclib, genius, whisper, etc.)
            duration_seconds: Duration of the song in seconds
            metadata: Additional metadata

        Returns:
            The created history entry
        """
        history = self._load_history()

        # Create the entry
        entry = {
            "id": len(history) + 1,
            "artist": artist or "Unknown Artist",
            "title": title or "Unknown Title",
            "timestamp": datetime.now().isoformat(),
            "audio_path": audio_path,
            "video_path": video_path,
            "lyrics_source": lyrics_source,
            "duration_seconds": duration_seconds,
            "metadata": metadata or {}
        }

        # Check if this song already exists (by artist + title)
        existing_idx = None
        for i, h in enumerate(history):
            if (h.get("artist", "").lower() == entry["artist"].lower() and
                h.get("title", "").lower() == entry["title"].lower()):
                existing_idx = i
                break

        if existing_idx is not None:
            # Update existing entry
            entry["id"] = history[existing_idx]["id"]
            history[existing_idx] = entry
        else:
            # Add new entry at the beginning
            history.insert(0, entry)

        # Trim to max size
        if len(history) > self.MAX_HISTORY_SIZE:
            history = history[:self.MAX_HISTORY_SIZE]

        self._save_history(history)
        logger.info(f"Added song to history: {artist} - {title}")

        return entry

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent song history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of history entries, most recent first
        """
        history = self._load_history()
        # Sort by timestamp descending
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return history[:limit]

    def get_song_by_id(self, song_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific song by its ID.

        Args:
            song_id: The song's ID

        Returns:
            The history entry or None if not found
        """
        history = self._load_history()
        for entry in history:
            if entry.get("id") == song_id:
                return entry
        return None

    def search_history(self, query: str) -> List[Dict[str, Any]]:
        """Search history by artist or title.

        Args:
            query: Search query

        Returns:
            Matching history entries
        """
        history = self._load_history()
        query_lower = query.lower()

        results = [
            entry for entry in history
            if (query_lower in entry.get("artist", "").lower() or
                query_lower in entry.get("title", "").lower())
        ]

        # Sort by timestamp descending
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results

    def clear_history(self):
        """Clear all history."""
        self._save_history([])
        logger.info("Song history cleared")

    def format_for_dropdown(self) -> List[tuple]:
        """Format history for use in a Gradio dropdown.

        Returns:
            List of (display_text, value) tuples
        """
        history = self.get_history(limit=20)
        choices = []

        for entry in history:
            artist = entry.get("artist", "Unknown")
            title = entry.get("title", "Unknown")
            date = entry.get("timestamp", "")[:10]  # Just the date part
            display = f"{artist} - {title} ({date})"
            value = json.dumps({"artist": artist, "title": title})
            choices.append((display, value))

        return choices


# Global instance
_history_manager = None


def get_history_manager() -> SongHistoryManager:
    """Get the global song history manager instance."""
    global _history_manager
    if _history_manager is None:
        _history_manager = SongHistoryManager()
    return _history_manager
