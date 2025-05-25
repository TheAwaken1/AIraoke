import logging
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError
import os
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def correct_lyrics(
        self,
        lyrics: str,
        prompt_template: str,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Correct lyrics using the LLM."""
        pass

    @abstractmethod
    def unload_model(self):
        """Unload the model to free memory."""
        pass

class LocalLLMProvider(LLMProvider):
    """
    A provider for running a local LLM using Hugging Face transformers.
    """
    def __init__(
        self,
        model: str = "Qwen/Qwen1.5-0.5B-Chat",
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.model_name = model
        self.model_path = os.path.join("models", model.replace("/", "_"))
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_model_downloaded(self, progress_callback=None):
        """Check if the model is downloaded, and download it if not."""
        try:
            if not os.path.exists(self.model_path):
                self.logger.info(f"Model {self.model_name} not found at {self.model_path}. Downloading...")
                # Avoid checking progress_callback in a way that triggers __len__
                if callable(progress_callback):
                    progress_callback(0.1, f"Downloading model {self.model_name}...")

                # Check if the repository exists
                hf_api = HfApi()
                try:
                    repo_info = hf_api.repo_info(repo_id=self.model_name, repo_type="model")
                    self.logger.info(f"Repository {self.model_name} found: {repo_info}")
                except RepositoryNotFoundError:
                    self.logger.error(f"Repository {self.model_name} not found on Hugging Face.")
                    raise

                # Attempt to download the model
                result = snapshot_download(
                    repo_id=self.model_name,
                    local_dir=self.model_path,
                    local_dir_use_symlinks=False,
                    cache_dir=self.model_path,
                )
                self.logger.info(f"Model {self.model_name} downloaded successfully to {self.model_path}. Result: {result}")
                if callable(progress_callback):
                    progress_callback(0.3, "Model download complete!")
            else:
                self.logger.info(f"Model {self.model_name} already exists at {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to download model {self.model_name}: {str(e)}", exc_info=True)
            raise

    def load_model(self):
        """Load the local LLM model and tokenizer."""
        try:
            self.logger.info(f"Loading local LLM model: {self.model_name} from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.logger.info(f"Loaded local LLM model on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load local LLM model: {str(e)}")
            raise

    def correct_lyrics(
        self,
        lyrics: str,
        prompt_template: str,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Correct lyrics using the local LLM."""
        try:
            if self.model is None or self.tokenizer is None:
                self.load_model()
            prompt = prompt_template.format(lyrics=lyrics)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
            )
            corrected_lyrics = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            corrected_lyrics = corrected_lyrics[len(prompt):].strip()
            self.logger.info("Lyrics correction successful")
            return corrected_lyrics
        except Exception as e:
            self.logger.error(f"Error during lyrics correction: {str(e)}")
            return lyrics

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is None:
            return
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info("Unloaded local LLM model")

class OpenAIProvider(LLMProvider):
    """
    A provider for OpenAI-compatible APIs.
    """
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            self.model = model
        except ImportError:
            self.logger.error("OpenAI library not installed. Please install it with `pip install openai`.")
            raise

    def correct_lyrics(
        self,
        lyrics: str,
        prompt_template: str,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Correct lyrics using the OpenAI API."""
        try:
            prompt = prompt_template.format(lyrics=lyrics)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for correcting song lyrics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            corrected_lyrics = response.choices[0].message.content.strip()
            self.logger.info("Lyrics correction successful")
            return corrected_lyrics
        except Exception as e:
            self.logger.error(f"Error during lyrics correction with OpenAI: {str(e)}")
            return lyrics

    def unload_model(self):
        """No-op for API-based providers."""
        self.logger.info("No model to unload for OpenAIProvider")