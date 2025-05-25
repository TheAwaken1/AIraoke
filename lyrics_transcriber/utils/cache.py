import json
import os
from pathlib import Path
import logging
from typing import Optional, Any, Union  # Added Union import

class Cache:
    """
    A simple disk-based cache for storing and retrieving results.
    """
    def __init__(
        self,
        cache_dir: Union[str, Path],
        cache_file: str = "cache.json",
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load the cache from disk, or initialize an empty cache if it doesn't exist."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Error loading cache from {self.cache_file}: {str(e)}")
            return {}

    def _save_cache(self):
        """Save the cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving cache to {self.cache_file}: {str(e)}")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache by key."""
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        """Store a value in the cache by key."""
        self.cache[key] = value
        self._save_cache()