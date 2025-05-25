import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from lyrics_transcriber.correction.handlers.base import GapCorrectionHandler
from lyrics_transcriber.correction.handlers.llm_providers import LLMProvider
from lyrics_transcriber.types import WordCorrection, GapSequence
from lyrics_transcriber.utils.cache import Cache

class LLMHandler(GapCorrectionHandler):
    """
    A correction handler that uses an LLM to correct lyrics gaps.
    """
    def __init__(
        self,
        provider: LLMProvider,
        name: str,
        cache_dir: Union[str, Path],
        logger: Optional[logging.Logger] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        super().__init__(logger=logger)
        self.provider = provider
        self.name = name
        self.cache = Cache(cache_dir=cache_dir, logger=logger)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.prompt_template = "Correct the following transcribed lyrics for grammar, spelling, and accuracy, ensuring they match the style of song lyrics:\n{lyrics}"

    def can_handle(self, gap: GapSequence, handler_data: Dict) -> Tuple[bool, Optional[Dict]]:
        """Determine if the handler can process the gap."""
        gap_text = " ".join(handler_data["word_map"][word_id].text for word_id in gap.transcribed_word_ids)
        if len(gap_text.strip()) == 0:
            self.logger.debug("Gap text is empty, cannot handle")
            return False, None
        return True, {"gap_text": gap_text}

    def handle(self, gap: GapSequence, handler_data: Dict) -> List[WordCorrection]:
        """Correct the gap using the LLM."""
        gap_text = handler_data["gap_text"]
        cache_key = f"llm_correction_{self.name}_{gap_text}"

        # Check cache
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.logger.debug(f"Using cached correction for gap: '{gap_text}'")
            return cached_result

        # Retry logic for correction
        for attempt in range(self.max_retries):
            try:
                corrected_text = self.provider.correct_lyrics(
                    lyrics=gap_text,
                    prompt_template=self.prompt_template,
                    max_new_tokens=500,
                    temperature=0.7,
                    top_p=0.9
                )
                break
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.logger.error("Max retries reached, skipping correction")
                    return []
                time.sleep(self.retry_delay)

        # Process the corrected text into word corrections
        original_words = [handler_data["word_map"][word_id].text for word_id in gap.transcribed_word_ids]
        corrected_words = corrected_text.split()

        if len(corrected_words) != len(original_words):
            self.logger.warning(f"Corrected word count ({len(corrected_words)}) does not match original ({len(original_words)})")
            return []

        corrections = []
        for i, (orig, corr) in enumerate(zip(original_words, corrected_words)):
            if orig != corr:
                corrections.append(WordCorrection(
                    word_id=gap.transcribed_word_ids[i],
                    original_word=orig,
                    corrected_word=corr,
                    original_position=i,
                    corrected_position=i,
                    confidence=0.95,  # Assume high confidence for LLM corrections
                    reason=f"Corrected by {self.name}",
                    handler_name=self.name
                ))

        # Cache the result
        self.cache.set(cache_key, corrections)
        return corrections