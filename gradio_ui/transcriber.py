"""
Core transcriber functionality for the Lyrics Transcriber Gradio UI
"""

import os
import sys
import logging
import tempfile
import json
from pathlib import Path
import time
import warnings
import subprocess
import platform
import shlex
import shutil
import librosa
import numpy as np
import stat
import torch
from gradio_ui.video_renderer import render_video_with_background
from lyrics_transcriber.output.subtitles import SubtitlesGenerator
from lyrics_transcriber.output.segment_resizer import SegmentResizer
from lyrics_transcriber.utils.word_utils import WordUtils

warnings.filterwarnings("ignore", category=UserWarning, module="whisper.timing")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.model")

logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    logger.info("Attempting to import WhisperTranscriber and WhisperConfig...")
    from lyrics_transcriber.transcribers.whisper import WhisperTranscriber, WhisperConfig
    logger.info("WhisperTranscriber and WhisperConfig imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import WhisperTranscriber or WhisperConfig: {str(e)}")
    raise

try:
    logger.info("Attempting to import OutputGenerator...")
    from lyrics_transcriber.output.generator import OutputGenerator
    logger.info("OutputGenerator imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import OutputGenerator: {str(e)}")
    raise

try:
    logger.info("Attempting to import OutputConfig...")
    from lyrics_transcriber.core.config import OutputConfig
    logger.info("OutputConfig imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import OutputConfig: {str(e)}")
    raise

try:
    logger.info("Attempting to import LyricsCorrector...")
    from lyrics_transcriber.correction.corrector import LyricsCorrector
    logger.info("LyricsCorrector imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import LyricsCorrector: {str(e)}")
    raise

try:
    logger.info("Attempting to import TranscriptionResult and TranscriptionData...")
    from lyrics_transcriber.types import TranscriptionResult, TranscriptionData, LyricsSegment, Word
    logger.info("TranscriptionResult and TranscriptionData imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import TranscriptionResult or TranscriptionData: {str(e)}")
    raise

try:
    logger.info("Attempting to import LRCLIB provider...")
    from lyrics_transcriber.lyrics.lrclib import LRCLIBProvider
    from lyrics_transcriber.lyrics.base_lyrics_provider import LyricsProviderConfig
    logger.info("LRCLIB provider imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import LRCLIB provider: {str(e)}")
    LRCLIBProvider = None
    LyricsProviderConfig = None

TRANSCRIBER_AVAILABLE = True
logger.info("All modules imported successfully. TRANSCRIBER_AVAILABLE set to True.")

def hex_to_rgb(hex_color: str, alpha: int = 255) -> str:
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"{r},{g},{b},{alpha}"

class LyricsTranscriberWrapper:
    def __init__(self, use_gpu=False, output_dir=None, model_size="turbo", video_background="video_1", font_color="255,255,255,255", resolution="1080p", progress_callback=None, use_llm_correction=False, llm_corrector_model=None, enable_separate_vocals=False, post_intro_skip_seconds=10, use_beat_effects=True, video_effect="None", use_input_video=False, video_dimmer=0, enable_countdown=True, enable_pitch_guide=False, vocal_volume=100, custom_background_path=None):
        self.use_gpu = use_gpu
        self.output_dir = output_dir or os.path.join(tempfile.gettempdir(), "lyrics_transcriber_output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.video_background = video_background
        self.font_color = font_color
        self.resolution = resolution
        self.model_size = model_size
        self.use_llm_correction = use_llm_correction
        self.llm_corrector_model = llm_corrector_model
        self.enable_separate_vocals = enable_separate_vocals
        self.post_intro_skip_seconds = post_intro_skip_seconds
        self.use_beat_effects = use_beat_effects
        self.video_effect = video_effect
        self.use_input_video = use_input_video
        self.video_dimmer = video_dimmer
        self.enable_countdown = enable_countdown
        self.enable_pitch_guide = enable_pitch_guide
        self.vocal_volume = vocal_volume  # 0-100, where 0=instrumental only, 100=full vocals
        self.custom_background_path = custom_background_path
        
        self.demo_mode = False # Set to True if needed for testing without models

        # Parse resolution and get properly scaled font/line_height
        (self.video_width, self.video_height), self.font_size, self.line_height = self._get_video_params(resolution)

    def separate_and_mix_vocals(self, audio_filepath: str, vocal_volume: int, progress_callback=None) -> str:
        """Separate vocals from audio and remix at specified volume level.

        Args:
            audio_filepath: Path to the audio file
            vocal_volume: 0-100 where 0=instrumental only, 100=full mix

        Returns:
            Path to the mixed audio file
        """
        if vocal_volume >= 100:
            logger.info("Vocal volume is 100%, skipping separation")
            return audio_filepath

        logger.info(f"Starting vocal separation with vocal_volume={vocal_volume}%")

        try:
            if progress_callback:
                progress_callback(0.3, f"Separating vocals with Demucs (this takes a few minutes)...")

            # Create output directory for separated stems
            stems_dir = os.path.join(self.output_dir, "stems")
            os.makedirs(stems_dir, exist_ok=True)

            audio_basename = os.path.splitext(os.path.basename(audio_filepath))[0]
            # Clean the basename for directory naming
            safe_basename = audio_basename.replace(" ", "_").replace("'", "").replace('"', '')

            # Check if we already have separated stems (cache)
            stems_path = os.path.join(stems_dir, "htdemucs", safe_basename)
            vocals_path = os.path.join(stems_path, "vocals.wav")
            no_vocals_path = os.path.join(stems_path, "no_vocals.wav")

            # If stems don't exist, run demucs
            if not os.path.exists(vocals_path) or not os.path.exists(no_vocals_path):
                logger.info("Running Demucs vocal separation...")

                # Try using demucs directly (it should be in the venv)
                try:
                    import demucs.separate
                    import sys
                    from gradio_ui.demucs_runner import ensure_torchaudio_save_backend
                    ensure_torchaudio_save_backend()

                    # Run demucs using the module
                    demucs_args = [
                        "-n", "htdemucs",
                        "--two-stems", "vocals",
                        "-o", stems_dir,
                        audio_filepath
                    ]

                    logger.info(f"Demucs args: {demucs_args}")

                    # Save original argv and replace
                    original_argv = sys.argv
                    sys.argv = ["demucs"] + demucs_args

                    try:
                        demucs.separate.main()
                    finally:
                        sys.argv = original_argv

                except ImportError:
                    logger.warning("Demucs module not found, trying subprocess...")
                    # Fallback to subprocess
                    import subprocess
                    import sys

                    python_exe = sys.executable
                    runner_script = os.path.join(os.path.dirname(__file__), "demucs_runner.py")
                    demucs_cmd = [
                        python_exe, runner_script,
                        "-n", "htdemucs",
                        "--two-stems", "vocals",
                        "-o", stems_dir,
                        audio_filepath
                    ]

                    logger.info(f"Running Demucs via subprocess: {' '.join(demucs_cmd)}")
                    result = subprocess.run(
                        demucs_cmd,
                        capture_output=True,
                        text=True,
                        timeout=900,  # 15 minutes max
                        cwd=self.output_dir
                    )

                    logger.info(f"Demucs stdout: {result.stdout}")
                    if result.returncode != 0:
                        logger.error(f"Demucs failed with code {result.returncode}: {result.stderr}")
                        return audio_filepath

                # Find the output - demucs creates folder with track name
                # Try multiple possible paths
                possible_paths = [
                    os.path.join(stems_dir, "htdemucs", audio_basename),
                    os.path.join(stems_dir, "htdemucs", safe_basename),
                    os.path.join(stems_dir, "htdemucs", Path(audio_filepath).stem),
                ]

                stems_path = None
                for p in possible_paths:
                    if os.path.exists(p):
                        stems_path = p
                        break

                if stems_path is None:
                    # List what demucs actually created
                    htdemucs_dir = os.path.join(stems_dir, "htdemucs")
                    if os.path.exists(htdemucs_dir):
                        created_dirs = os.listdir(htdemucs_dir)
                        logger.info(f"Demucs created directories: {created_dirs}")
                        if created_dirs:
                            stems_path = os.path.join(htdemucs_dir, created_dirs[0])

                if stems_path is None:
                    logger.error("Could not find Demucs output")
                    return audio_filepath

                vocals_path = os.path.join(stems_path, "vocals.wav")
                no_vocals_path = os.path.join(stems_path, "no_vocals.wav")

            if not os.path.exists(vocals_path) or not os.path.exists(no_vocals_path):
                logger.error(f"Demucs output not found. Vocals: {os.path.exists(vocals_path)}, No-vocals: {os.path.exists(no_vocals_path)}")
                return audio_filepath

            logger.info(f"Found stems - Vocals: {vocals_path}, Instrumental: {no_vocals_path}")

            if progress_callback:
                progress_callback(0.7, "Mixing vocals at desired level...")

            # Mix vocals at desired volume using pydub
            from pydub import AudioSegment

            vocals = AudioSegment.from_wav(vocals_path)
            instrumental = AudioSegment.from_wav(no_vocals_path)

            # Calculate dB reduction for vocals
            if vocal_volume == 0:
                # Pure instrumental
                mixed = instrumental
                logger.info("Using pure instrumental (0% vocals)")
            else:
                # Reduce vocal volume using logarithmic scale
                # 100% = 0dB, 50% = -6dB, 25% = -12dB, 0% = -inf
                db_reduction = 20 * np.log10(vocal_volume / 100) if vocal_volume > 0 else -60
                logger.info(f"Reducing vocals by {db_reduction:.1f} dB")
                vocals_adjusted = vocals + db_reduction
                mixed = instrumental.overlay(vocals_adjusted)

            # Export mixed audio
            output_path = os.path.join(self.output_dir, f"{safe_basename}_mixed_{vocal_volume}pct.wav")
            mixed.export(output_path, format="wav")

            if progress_callback:
                progress_callback(0.85, "Vocal mixing complete!")

            logger.info(f"Created mixed audio with {vocal_volume}% vocals: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error in vocal separation/mixing: {e}", exc_info=True)
            return audio_filepath  # Fall back to original

    def fetch_lyrics(self, artist: str, title: str, provider: str = "lrclib") -> dict:
        """Fetch lyrics from the specified provider.

        Args:
            artist: Artist name
            title: Song title
            provider: Provider name (lrclib, genius, spotify)

        Returns:
            Dict with success, lyrics_text, is_synced, segments, and metadata
        """
        if not artist or not title:
            return {
                "success": False,
                "message": "Please provide both artist and song title",
                "lyrics_text": None,
                "is_synced": False,
                "segments": [],
                "metadata": {}
            }

        try:
            if provider == "lrclib" and LRCLIBProvider:
                config = LyricsProviderConfig(
                    cache_dir=os.path.join(tempfile.gettempdir(), "lyrics_cache")
                )
                lrclib = LRCLIBProvider(config=config, logger=logger)
                result = lrclib.fetch_lyrics(artist, title)

                if result:
                    lyrics_text = result.get_full_text()
                    is_synced = result.metadata.is_synced

                    return {
                        "success": True,
                        "message": f"Found {'synced' if is_synced else 'plain'} lyrics from LRCLIB",
                        "lyrics_text": lyrics_text,
                        "is_synced": is_synced,
                        "segments": result.segments,
                        "metadata": {
                            "source": result.source,
                            "track_name": result.metadata.track_name,
                            "artist_names": result.metadata.artist_names,
                            "album_name": result.metadata.album_name,
                            "duration_ms": result.metadata.duration_ms,
                        }
                    }
                else:
                    return {
                        "success": False,
                        "message": f"No lyrics found on LRCLIB for {artist} - {title}",
                        "lyrics_text": None,
                        "is_synced": False,
                        "segments": [],
                        "metadata": {}
                    }
            else:
                return {
                    "success": False,
                    "message": f"Provider '{provider}' not available",
                    "lyrics_text": None,
                    "is_synced": False,
                    "segments": [],
                    "metadata": {}
                }

        except Exception as e:
            logger.error(f"Error fetching lyrics: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Error fetching lyrics: {str(e)}",
                "lyrics_text": None,
                "is_synced": False,
                "segments": [],
                "metadata": {}
            }

    def transcribe_audio(self, audio_filepath, progress_callback=None):
        if self.demo_mode:
            # In demo mode, we simulate the structure
            result = self._demo_transcribe(audio_filepath, progress_callback=progress_callback)
            # Create dummy segments from the demo text for consistency if needed, 
            # but _demo_transcribe returns a dict that might not match exactly.
            # For now, let's just return what _demo_transcribe returns but ensuring keys exist.
            return result

        try:
            if progress_callback: progress_callback(0.1, "Initializing Whisper...")

            # 1. WHISPER TRANSCRIPTION
            logger.info(f"Transcribing {audio_filepath}...")
            transcriber = WhisperTranscriber(
                cache_dir=os.path.join(tempfile.gettempdir(), "whisper_cache")
            )
            os.environ["WHISPER_MODEL"] = self.model_size
            os.environ["WHISPER_DEVICE"] = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"

            transcription_result = transcriber.transcribe(audio_filepath)
            if progress_callback: progress_callback(0.4, "Transcription done.")

            # 2. BEAT DETECTION
            beat_times = []
            if self.use_beat_effects:
                if progress_callback: progress_callback(0.6, "Detecting beats...")
                try:
                    y, sr = librosa.load(audio_filepath, sr=None)
                    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
                except Exception as e:
                    logger.error(f"Beat detection failed: {e}")

            # 3. CONVERT SEGMENTS
            if progress_callback: progress_callback(0.7, "Processing lyrics...")
            segments = []
            for seg in transcription_result.segments:
                lyrics_seg = self._dict_to_lyrics_segment(seg)
                if lyrics_seg:
                    segments.append(lyrics_seg)

            lyrics_text = "\n".join([seg.text for seg in segments])

            return {
                "success": True,
                "segments": segments,
                "beat_times": beat_times,
                "lyrics_text": lyrics_text
            }
        except Exception as e:
            logger.error(f"Error in transcription audio phase: {e}", exc_info=True)
            return {"success": False, "message": str(e)}

    def render_video(self, audio_filepath, segments, beat_times, artist=None, title=None, progress_callback=None):
        try:
            # Process vocals if needed
            final_audio_path = audio_filepath
            if self.vocal_volume < 100:
                if progress_callback: progress_callback(0.5, f"Adjusting vocals to {self.vocal_volume}%...")
                final_audio_path = self.separate_and_mix_vocals(
                    audio_filepath,
                    self.vocal_volume,
                    progress_callback
                )

            if progress_callback: progress_callback(0.8, "Rendering video...")
            
            # 4. CREATE STYLES.JSON IF MISSING
            styles_path = os.path.join(os.path.dirname(__file__), "styles.json")
            if not os.path.exists(styles_path):
                default_styles = {
                    "default": {
                        "font_name": "Arial",
                        "font_size": 54,
                        "primary_colour": "&H00FFFFFF",
                        "secondary_colour": "&H00FFFFFF",
                        "outline_colour": "&H00000000",
                        "back_colour": "&H00000000",
                        "bold": True,
                        "italic": False,
                        "underline": False,
                        "strikeout": False,
                        "scale_x": 100,
                        "scale_y": 100,
                        "spacing": 0,
                        "angle": 0,
                        "border_style": 1,
                        "outline": 2,
                        "shadow": 2,
                        "alignment": 2,
                        "margin_l": 10,
                        "margin_r": 10,
                        "margin_v": 40
                    }
                }
                with open(styles_path, "w", encoding="utf-8") as f:
                    json.dump(default_styles, f, indent=2)

            # 5. GENERATE ASS SUBTITLES
            ass_generator = SubtitlesGenerator(
                output_dir=self.output_dir,
                video_resolution=(self.video_width, self.video_height),
                font_size=self.font_size,
                line_height=self.line_height,
                styles=self._prepare_ass_styles()
            )
            song_name = title or os.path.splitext(os.path.basename(audio_filepath))[0]
            ass_file = ass_generator.generate_ass(
                segments,
                output_prefix=song_name,
                audio_filepath=audio_filepath
            )

            # 6. RENDER VIDEO
            video_path_to_return = None
            if os.path.exists(ass_file):
                output_video_path = os.path.join(self.output_dir, f"{song_name}_karaoke.mp4")

                # Detect first vocal time for countdown
                first_vocal_time = None
                if segments and len(segments) > 0:
                    first_vocal_time = segments[0].start_time
                    logger.info(f"First vocal detected at {first_vocal_time:.2f}s")

                rendered_video_path = render_video_with_background(
                    audio_filepath=final_audio_path,
                    ass_filepath=ass_file,
                    output_filepath=output_video_path,
                    background_image=self._create_background_image(),
                    resolution=self.resolution,
                    beat_times=beat_times,
                    video_effect=self.video_effect,
                    use_input_video=self.use_input_video,
                    original_video_path=audio_filepath if self.use_input_video else None,
                    video_dimmer=self.video_dimmer,
                    enable_countdown=self.enable_countdown,
                    first_vocal_time=first_vocal_time,
                    enable_pitch_guide=self.enable_pitch_guide
                )
                if rendered_video_path and os.path.exists(rendered_video_path):
                    video_path_to_return = rendered_video_path

            if progress_callback: progress_callback(1.0, "Complete!")
            
            return {
                "success": True,
                "video_path": video_path_to_return
            }

        except Exception as e:
            logger.error(f"Error in render video phase: {e}", exc_info=True)
            return {"success": False, "message": str(e)}

    def transcribe(self, audio_filepath, artist=None, title=None, progress_callback=None):
        # Phase 1: Transcribe
        transcription_result = self.transcribe_audio(audio_filepath, progress_callback)
        if not transcription_result.get("success"):
            return transcription_result
        
        # Phase 2: Render
        render_result = self.render_video(
            audio_filepath, 
            transcription_result["segments"], 
            transcription_result.get("beat_times", []), 
            artist, 
            title, 
            progress_callback
        )
        
        # Merge results
        return {
            "success": render_result.get("success"),
            "lyrics_text": transcription_result.get("lyrics_text"),
            "video_path": render_result.get("video_path"),
            "message": render_result.get("message")
        }

    def _create_background_image(self):
        backgrounds_dir = os.path.join(os.path.dirname(__file__), "backgrounds")
        logger.debug(f"Checking background for video_background: {self.video_background}")
        
        if self.video_background == "custom":
            if self.custom_background_path and os.path.exists(self.custom_background_path):
                logger.info(f"Using custom background video: {self.custom_background_path}")
                return self.custom_background_path
            logger.warning("Custom background selected but no file provided. Using black background.")
            return None
        elif self.video_background == "black":
            logger.info("Using black background.")
            return None
        elif self.video_background == "audio_particles":
            logger.info("Using audio particle visualization.")
            return "audio_particles"
        
        # Check for video files
        bg_path = os.path.join(backgrounds_dir, f"{self.video_background}.mp4")
        if os.path.exists(bg_path):
            logger.info(f"Using pre-rendered background: {bg_path}")
            return bg_path
            
        # Fallback
        logger.warning(f"Background {self.video_background} not found. Using black background.")
        return None

    def _get_resolution_dimensions(self, resolution: str) -> tuple:
        resolution_map = {
            "360p": (640, 360),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "4k": (3840, 2160)
        }
        if resolution not in resolution_map:
            logger.warning(f"Invalid resolution '{resolution}'. Defaulting to 1080p.")
            resolution = "1080p"
        return resolution_map[resolution]

    def rgba_to_ass_color(self, rgba_string: str) -> str:
        try:
            r, g, b, a = map(int, rgba_string.split(','))
            ass_alpha = f"{255 - a:02X}"
            ass_blue = f"{b:02X}"
            ass_green = f"{g:02X}"
            ass_red = f"{r:02X}"
            return f"&H{ass_alpha}{ass_blue}{ass_green}{ass_red}"
        except Exception as e:
            logger.error(f"Color conversion failed for '{rgba_string}': {e}. Defaulting to white.")
            return "&H00FFFFFF"

    def _get_video_params(self, resolution: str) -> tuple:
        resolution_map = {
            "360p": (640, 360),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "4k": (3840, 2160)
        }

        # Resolution-specific settings to prevent lyrics overlap
        # Each entry: (font_size, line_height) - line_height should be ~1.8x font_size
        resolution_settings = {
            "360p": (24, 44),    # Small text for low res
            "720p": (40, 72),    # Base settings
            "1080p": (54, 98),   # Scaled up for HD
            "4k": (96, 172)      # Large text for 4K
        }

        if resolution not in resolution_map:
            logger.warning(f"Invalid resolution '{resolution}'. Defaulting to 1080p.")
            resolution = "1080p"

        dims = resolution_map[resolution]
        font_size, line_height = resolution_settings[resolution]

        logger.debug(f"Video params for {resolution}: dims={dims}, font_size={font_size}, line_height={line_height}")
        return dims, font_size, line_height

    def _prepare_ass_styles(self) -> dict:
        karaoke_highlight_fill_rgba = self.font_color
        karaoke_base_fill_rgba = "255,255,255,255"
        karaoke_outline_rgba = "0,0,0,255"
        karaoke_shadow_rgba = "0,0,0,255"

        styles = {
            "enable_ass": True,
            "karaoke": {
                "ass_name": "Default",
                "font_size": str(self.font_size),
                "font": "Arial",
                "font_path": "C:/Windows/Fonts/arial.ttf",
                "primary_color": karaoke_highlight_fill_rgba,
                "secondary_color": karaoke_base_fill_rgba,
                "outline_color": karaoke_outline_rgba,
                "back_color": karaoke_shadow_rgba,
                "bold": "0",
                "italic": "0",
                "underline": "0",
                "strike_out": "0",
                "scale_x": "100",
                "scale_y": "100",
                "spacing": "0",
                "angle": "0",
                "border_style": "1",
                "outline": "0",
                "shadow": "1",
                "alignment": "2",
                "margin_l": "10",
                "margin_r": "10",
                "margin_v": "20",
                "encoding": "1",
            },
            "karaoke_active": {
                "ass_name": "Active",
                "font_size": str(self.font_size),
                "font": "Arial",
                "font_path": "C:/Windows/Fonts/arial.ttf",
                "primary_color": karaoke_highlight_fill_rgba,
                "secondary_color": karaoke_base_fill_rgba,
                "outline_color": karaoke_outline_rgba,
                "back_color": karaoke_shadow_rgba,
                "bold": "1",
                "italic": "0",
                "underline": "0",
                "strike_out": "0",
                "scale_x": "100",
                "scale_y": "100",
                "spacing": "0",
                "angle": "0",
                "border_style": "1",
                "outline": "0",
                "shadow": "2",
                "alignment": "2",
                "margin_l": "10",
                "margin_r": "10",
                "margin_v": "20",
                "encoding": "1",
            }
        }
        logger.debug(
            f"Prepared ASS styles (using RGBA strings for subtitles.py). "
            f"Style 'karaoke' (Default): Primary(HighlightFill)='{karaoke_highlight_fill_rgba}', "
            f"Secondary(BaseFill)='{karaoke_base_fill_rgba}', Outline='{karaoke_outline_rgba}'. "
            f"Style 'karaoke_active': Primary='{karaoke_highlight_fill_rgba}', Outline='{karaoke_outline_rgba}'. Fontsize: {self.font_size}"
        )
        return styles

    def _dict_to_lyrics_segment(self, segment_dict, words=None):
        if not isinstance(segment_dict, dict):
            return segment_dict

        segment_text = segment_dict.get('text', '').strip()
        if not segment_text:
            logger.debug(f"Skipping segment with empty text: {segment_dict}")
            return None

        segment_words = []
        if words and 'start_time' in segment_dict and 'end_time' in segment_dict:
            start_time = segment_dict['start_time']
            end_time = segment_dict['end_time']
            segment_words = [
                word for word in words
                if start_time <= word.start_time < end_time and word.confidence > 0.1
            ]

        duration = segment_dict.get('end_time', 0.0) - segment_dict.get('start_time', 0.0)
        if not segment_words or duration < 0.5:
            logger.debug(f"Skipping segment with no valid words or short duration: {segment_dict}")
            return None

        if segment_words:
            avg_confidence = sum(word.confidence for word in segment_words) / len(segment_words)
            if avg_confidence < 0.2:
                logger.debug(f"Skipping segment with low average confidence ({avg_confidence}): {segment_dict}")
                return None
            logger.debug(f"Segment words confidences: {[f'{word.text}: {word.confidence}' for word in segment_words]}")

        return LyricsSegment(
            id=segment_dict.get('id', WordUtils.generate_id()),
            text=segment_text,
            words=segment_words,
            start_time=segment_dict.get('start', segment_dict.get('start_time', 0.0)),
            end_time=segment_dict.get('end', segment_dict.get('end_time', 0.0)),
        )

    def _get_segment_start_time(self, segment):
        if isinstance(segment, dict):
            return segment.get('start', segment.get('start_time', 0.0))
        return getattr(segment, 'start', getattr(segment, 'start_time', 0.0))

    def _format_lyrics_from_segments(self, segments):
        if not segments:
            return "No lyrics segments available"
        
        formatted_text = ""
        for segment in segments:
            start_time = self._format_time(self._get_segment_start_time(segment))
            end_time = self._format_time(segment.end_time if hasattr(segment, 'end_time') else segment['end_time'])
            text = segment.text if hasattr(segment, 'text') else segment['text']
            formatted_text += f"[{start_time} - {end_time}] {text}\n"
        
        return formatted_text
    
    def _format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:05.2f}"
    
    def _demo_transcribe(self, audio_filepath, artist=None, title=None, progress_callback=None):
        try:
            if not os.path.exists(audio_filepath):
                return {
                    "success": False,
                    "message": f"File not found: {audio_filepath}",
                    "lyrics_text": None,
                    "download_files": [],
                    "results": None,
                    "video_path": None
                }
            
            file_size = os.path.getsize(audio_filepath)
            file_name = os.path.basename(audio_filepath)
            
            if progress_callback:
                progress_callback(0.1, "Initializing transcriber...")
                time.sleep(1)
                progress_callback(0.3, "Transcribing audio...")
                time.sleep(2)
                progress_callback(0.6, "Processing lyrics...")
                time.sleep(1)
                progress_callback(0.8, "Generating output files...")
                time.sleep(1)
                progress_callback(1.0, "Transcription complete!")
            
            artist_name = artist or "Unknown Artist"
            song_name = title or os.path.splitext(file_name)[0]
            
            output_dir = self.output_dir
            lrc_file = os.path.join(output_dir, f"{song_name}.lrc")
            ass_file = os.path.join(output_dir, f"{song_name}.ass")
            txt_file = os.path.join(output_dir, f"{song_name}.txt")
            
            demo_lyrics = self._generate_demo_lyrics(artist_name, song_name)
            
            with open(lrc_file, 'w') as f:
                f.write(self._generate_demo_lrc(artist_name, song_name))
            
            with open(ass_file, 'w') as f:
                f.write(self._generate_demo_ass(artist_name, song_name))
                
            with open(txt_file, 'w') as f:
                f.write(demo_lyrics)
            
            return {
                "success": True,
                "message": "Demo transcription completed successfully",
                "lyrics_text": demo_lyrics,
                "download_files": [lrc_file, ass_file, txt_file],
                "results": None,
                "video_path": None
            }
            
        except Exception as e:
            logger.error(f"Error in demo transcription: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Error in demo transcription: {str(e)}",
                "lyrics_text": None,
                "download_files": [],
                "results": None,
                "video_path": None
            }
    
    def _generate_demo_lyrics(self, artist, title):
        return f"""[00:00.00 - 00:05.00] {title} by {artist}
[00:05.00 - 00:10.00] This is a demo transcription
[00:10.00 - 00:15.00] The actual transcription would use the Python Lyrics Transcriber
[00:15.00 - 00:20.00] With word-level timestamps and synchronization
[00:20.00 - 00:25.00] Perfect for karaoke video production
[00:25.00 - 00:30.00] This is just a placeholder for demonstration
[00:30.00 - 00:35.00] In the real version, lyrics would be extracted from the audio
[00:35.00 - 00:40.00] And matched with online lyrics if available
[00:40.00 - 00:45.00] Thank you for trying the Lyrics Transcriber!
"""
    
    def _generate_demo_lrc(self, artist, title):
        return f"""[ar:{artist}]
[ti:{title}]
[al:Demo Album]
[by:Lyrics Transcriber]
[00:00.00]
[00:05.00]{title} by {artist}
[00:10.00]This is a demo transcription
[00:15.00]The actual transcription would use the Python Lyrics Transcriber
[00:20.00]With word-level timestamps and synchronization
[00:25.00]Perfect for karaoke video production
[00:30.00]This is just a placeholder for demonstration
[00:35.00]In the real version, lyrics would be extracted from the audio file
[00:40.00]And matched with online lyrics if available
[00:45.00]Thank you for trying the Lyrics Transcriber!
"""
    
    def _generate_demo_ass(self, artist, title):
        return f"""[Script Info]
Title: {title}
Artist: {artist}
ScriptType: v4.00+
PlayResX: {self.video_width}
PlayResY: {self.video_height}
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{self.font_size},{self.rgba_to_ass_color(self.font_color)},&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:05.00,0:00:10.00,Default,,0,0,0,,{title} by {artist}
Dialogue: 0,0:00:10.00,0:00:15.00,Default,,0,0,0,,This is a demo transcription
Dialogue: 0,0:00:15.00,0:00:20.00,Default,,0,0,0,,The actual transcription would use the Python Lyrics Transcriber
Dialogue: 0,0:00:20.00,0:00:25.00,Default,,0,0,0,,With word-level timestamps and synchronization
Dialogue: 0,0:00:25.00,0:00:30.00,Default,,0,0,0,,Perfect for karaoke video production
Dialogue: 0,0:00:30.00,0:00:35.00,Default,,0,0,0,,This is just a placeholder for demonstration
Dialogue: 0,0:00:35.00,0:00:40.00,Default,,0,0,0,,In the real version, lyrics would be extracted from the audio
Dialogue: 0,0:00:40.00,0:00:45.00,Default,,0,0,0,,And matched with online lyrics if available
Dialogue: 0,0:00:45.00,0:00:50.00,Default,,0,0,0,,Thank you for trying the Lyrics Transcriber!
"""