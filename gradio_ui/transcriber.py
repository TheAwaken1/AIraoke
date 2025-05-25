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

TRANSCRIBER_AVAILABLE = True
logger.info("All modules imported successfully. TRANSCRIBER_AVAILABLE set to True.")

def hex_to_rgb(hex_color: str, alpha: int = 255) -> str:
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"{r},{g},{b},{alpha}"

class LyricsTranscriberWrapper:
    def __init__(self, use_gpu=False, output_dir=None, model_size="turbo", video_background="video_1", font_color="255,255,255,255", resolution="1080p", progress_callback=None, use_llm_correction=False, enable_separate_vocals=False, post_intro_skip_seconds=10, use_beat_effects=True):
        self.use_gpu = use_gpu
        self.output_dir = output_dir or os.path.join(tempfile.gettempdir(), "lyrics_transcriber_output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.video_background = video_background
        self.font_color = font_color
        self.resolution = resolution
        self.progress_callback = progress_callback
        self.background_image = None
        self.demo_mode = not TRANSCRIBER_AVAILABLE
        self.use_llm_correction = use_llm_correction
        self.enable_separate_vocals = enable_separate_vocals
        self.post_intro_skip_seconds = max(0, post_intro_skip_seconds)
        self.use_beat_effects = use_beat_effects

        if self.demo_mode:
            logger.warning("Lyrics Transcriber modules not available. Running in demo mode.")
            return

        if self.use_gpu:
            try:
                from gradio_ui.gpu_utils import check_cuda_availability, configure_torch_for_gpu
                has_gpu, gpu_info, vram_gb = check_cuda_availability()
                if has_gpu:
                    logger.info(f"GPU detected: {gpu_info}")
                    configure_torch_for_gpu(vram_gb)
                else:
                    logger.warning(f"GPU requested but not available: {gpu_info}")
                    self.use_gpu = False
            except Exception as e:
                logger.error(f"Error configuring GPU: {e}")
                self.use_gpu = False

        cache_dir = os.path.join(self.output_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        self.video_width, self.video_height = self._get_resolution_dimensions(self.resolution)
        self.video_resolution_num, self.font_size, self.line_height = self._get_video_params(self.resolution)
        self.background_image = self._create_background_image()

        output_styles = {
            "enable_ass": True,
            "enable_lrc": True,
            "enable_review": False,
            "enable_txt": True,
            "enable_json": True,
            "enable_cdg": False,
            "karaoke": {
                "ass_name": "Default",
                "font_size": str(self.font_size),
                "top_padding": 50,
                "font": "Arial",
                "font_path": "C:/Windows/Fonts/arial.ttf",
                "primary_color": "255,255,255,255",
                "secondary_color": "0,0,0,255",
                "outline_color": "255,255,255,255",
                "back_color": "0,0,0,0",
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
                "subtitle_offset_ms": 0
            },
            "karaoke_active": {
                "ass_name": "Active",
                "font_size": str(self.font_size),
                "top_padding": 50,
                "font": "Arial",
                "font_path": "C:/Windows/Fonts/arial.ttf",
                "primary_color": self.font_color,
                "secondary_color": "255,105,180,255",
                "outline_color": "255,255,255,255",
                "back_color": "0,0,0,0",
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
                "encoding": "1"
            }
        }
        output_styles_json_path = os.path.join(self.output_dir, "output_styles.json")
        with open(output_styles_json_path, 'w') as f:
            json.dump(output_styles, f)
        logger.info(f"Font set to: {output_styles['karaoke']['font']}, size: {self.font_size}")
        logger.debug(f"Font color set to: {self.font_color} (highlight), base color set to white (255,255,255,255)")

        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        self.transcriber = WhisperTranscriber(
            cache_dir=cache_dir,
            config=WhisperConfig(model_size=model_size),
            logger=logger
        )
        if self.use_llm_correction:
            self.corrector = LyricsCorrector(
                cache_dir=cache_dir,
                logger=logger,
                progress_callback=self.progress_callback
            )
            logger.info("LyricsCorrector initialized for LLM correction.")
        else:
            self.corrector = None
            logger.info("LLM correction disabled. LyricsCorrector not initialized.")

        self.output_config = OutputConfig(
            output_styles_json=output_styles_json_path,
            output_dir=self.output_dir,
            cache_dir=os.path.join(self.output_dir, "cache"),
            fetch_lyrics=False,
            run_transcription=True,
            run_correction=self.use_llm_correction,
            enable_review=False,
            max_line_length=36,
            subtitle_offset_ms=0,
            render_video=False,
            generate_cdg=False,
            generate_plain_text=True,
            generate_lrc=True,
            video_resolution=self.resolution
        )
        self.output_generator = OutputGenerator(
            config=self.output_config
        )
        self.video_resolution_num, self.font_size, self.line_height = self._get_video_params(self.resolution)
        ass_styles = self._prepare_ass_styles()
        self.subtitle_generator = SubtitlesGenerator(
            output_dir=self.output_config.output_dir,
            video_resolution=self.video_resolution_num,
            font_size=self.font_size,
            line_height=self.line_height,
            styles=ass_styles,
            subtitle_offset_ms=self.output_config.subtitle_offset_ms,
            logger=logger
        )
        try:
            self.segment_resizer = SegmentResizer(
                max_line_length=self.output_config.max_line_length,
                logger=logger
            )
            logger.info("SegmentResizer initialized successfully.")
        except NameError:
            logger.error("SegmentResizer class not imported correctly. Skipping initialization.")
            self.segment_resizer = None
        except Exception as e:
            logger.error(f"Failed to initialize SegmentResizer: {e}", exc_info=True)
            self.segment_resizer = None
        self.corrected_lyrics = None
        self.beat_times = []  # Store beat timestamps

    def detect_beats(self, audio_filepath):
        """Detect beat timestamps in an audio file using librosa."""
        logger.info(f"Detecting beats for: {audio_filepath}")
        try:
            y, sr = librosa.load(audio_filepath)
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            logger.info(f"Detected {len(beat_times)} beats at tempo {tempo.item():.2f} BPM")
            logger.debug(f"Beat timestamps: {beat_times.tolist()}")
            self.beat_times = beat_times.tolist()
            return self.beat_times, tempo.item()
        except Exception as e:
            logger.error(f"Beat detection failed: {str(e)}", exc_info=True)
            self.beat_times = []
            return [], None

    def separate_vocals(self, audio_filepath, output_prefix):
        """Separate vocals from audio using Demucs and return path to instrumental audio."""
        logger.debug(f"Starting vocal separation for: {audio_filepath}")
        
        if not shutil.which("demucs"):
            logger.error("Demucs executable not found in PATH")
            raise RuntimeError("Demucs is not installed or not found in PATH. Please install Demucs.")

        try:
            demucs_model = "mdx_extra_q"
            try:
                import diffq
                logger.info("diffq package found, using mdx_extra_q Demucs model.")
                model_dir = "mdx_extra_q"
            except ImportError:
                logger.warning("diffq package not found, falling back to htdemucs model. Install diffq with: python.exe -m pip install diffq")
                demucs_model = "htdemucs"
                model_dir = "htdemucs"
            
            separation_dir = os.path.join(self.output_dir, model_dir, output_prefix)
            instrumental_wav = os.path.join(separation_dir, "no_vocals.wav")
            instrumental_mp3 = os.path.join(self.output_dir, f"{output_prefix}_instrumental.mp3")
            
            audio_filepath = os.path.normpath(audio_filepath)
            separation_dir = os.path.normpath(separation_dir)
            instrumental_wav = os.path.normpath(instrumental_wav)
            instrumental_mp3 = os.path.normpath(instrumental_mp3)
            
            if os.path.exists(instrumental_mp3):
                logger.info(f"Instrumental audio already exists: {instrumental_mp3}")
                return instrumental_mp3
            
            os.makedirs(separation_dir, exist_ok=True)
            
            cmd = [
                "demucs",
                "-n", demucs_model,
                "--two-stems=vocals",
                "--out", os.path.normpath(self.output_dir),
                audio_filepath
            ]
            logger.debug(f"Running Demucs command: {' '.join(shlex.quote(str(c)) for c in cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600
            )
            
            if result.returncode != 0:
                logger.error(f"Demucs failed with return code {result.returncode}")
                logger.error(f"Stderr: {result.stderr}")
                logger.error(f"Stdout: {result.stdout}")
                raise RuntimeError(f"Demucs vocal separation failed: {result.stderr}")
            
            logger.info(f"Demucs output: {result.stdout}")
            
            logger.debug(f"Checking separation directory: {separation_dir}")
            if os.path.exists(separation_dir):
                logger.debug(f"Files in {separation_dir}: {os.listdir(separation_dir)}")
                possible_instrumental = os.path.join(separation_dir, "instrumental.wav")
                if not os.path.exists(instrumental_wav) and os.path.exists(possible_instrumental):
                    logger.info(f"Found alternative instrumental file: {possible_instrumental}")
                    instrumental_wav = possible_instrumental
                elif not os.path.exists(instrumental_wav):
                    logger.error(f"Instrumental audio not found at {instrumental_wav}")
                    raise FileNotFoundError(f"Demucs did not produce instrumental audio at {instrumental_wav}")
            else:
                logger.error(f"Separation directory not found: {separation_dir}")
                raise FileNotFoundError(f"Demucs did not create output directory: {separation_dir}")
            
            binary_name = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
            app_base = os.path.dirname(os.path.dirname(__file__))
            possible_paths = [
                os.path.join(app_base, "bin", binary_name),
                os.path.join(app_base, "ffmpeg", binary_name),
                os.path.join(app_base, binary_name),
            ]
            
            ffmpeg_path = None
            for path in possible_paths:
                logger.debug(f"Checking FFmpeg path: {path}")
                if os.path.exists(path):
                    try:
                        result = subprocess.run([path, "-version"], capture_output=True, text=True, check=True)
                        logger.debug(f"FFmpeg version: {result.stdout.splitlines()[0]}")
                        ffmpeg_path = path
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        logger.warning(f"FFmpeg failed at {path}: {e}")
            
            if not ffmpeg_path:
                ffmpeg_path = "ffmpeg"
                try:
                    result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, check=True)
                    logger.debug(f"System PATH FFmpeg version: {result.stdout.splitlines()[0]}")
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    logger.error(f"System PATH FFmpeg not found or failed: {e}")
                    raise RuntimeError("FFmpeg not found in app folder or system PATH.")
            
            logger.debug(f"Using ffmpeg path: {ffmpeg_path}")
            if os.path.exists(ffmpeg_path):
                try:
                    file_stats = os.stat(ffmpeg_path)
                    logger.debug(f"ffmpeg permissions: {oct(file_stats.st_mode & 0o777)}")
                    if not (file_stats.st_mode & stat.S_IXUSR):
                        logger.warning(f"ffmpeg at {ffmpeg_path} is not executable")
                except Exception as e:
                    logger.warning(f"Could not check ffmpeg file stats: {e}")
            
            ffmpeg_cmd = [
                ffmpeg_path, "-y",
                "-i", instrumental_wav,
                "-c:a", "mp3",
                "-b:a", "192k",
                instrumental_mp3
            ]
            logger.debug(f"Running FFmpeg command: {' '.join(shlex.quote(str(c)) for c in ffmpeg_cmd)}")
            ffmpeg_result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300
            )
            
            if ffmpeg_result.returncode != 0:
                logger.error(f"FFmpeg conversion failed with return code {ffmpeg_result.returncode}")
                logger.error(f"Stderr: {ffmpeg_result.stderr}")
                logger.error(f"Stdout: {ffmpeg_result.stdout}")
                raise RuntimeError(f"FFmpeg conversion failed: {ffmpeg_result.stderr}")
            
            if not os.path.exists(instrumental_mp3):
                logger.error(f"Instrumental MP3 not found at {instrumental_mp3}")
                raise FileNotFoundError(f"FFmpeg did not produce instrumental MP3 at {instrumental_mp3}")
            
            logger.info(f"Successfully generated instrumental audio: {instrumental_mp3}")
            return instrumental_mp3
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"Demucs or FFmpeg timed out after {e.timeout} seconds")
            raise RuntimeError(f"Vocal separation timed out: {str(e)}")
        except FileNotFoundError as e:
            logger.error(f"Demucs or FFmpeg error: {str(e)}")
            raise RuntimeError(f"Required file not found: {str(e)}")
        except Exception as e:
            logger.error(f"Error separating vocals: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to separate vocals: {str(e)}")

    # In _create_background_image method:
    def _create_background_image(self):
        backgrounds_dir = os.path.join(os.path.dirname(__file__), "backgrounds")
        logger.debug(f"Checking background for video_background: {self.video_background}")
        
        if self.video_background == "black":
            logger.info("Using black background.")
            return None
        elif self.video_background == "audio_particles":
            logger.info("Using audio particle visualization.")
            return "audio_particles"  # Return special string, not a file path
        elif self.video_background == "video_1":
            bg_path = os.path.join(backgrounds_dir, "video_1.mp4")
        elif self.video_background == "video_2":
            bg_path = os.path.join(backgrounds_dir, "video_2.mp4")
        elif self.video_background == "video_3":
            bg_path = os.path.join(backgrounds_dir, "video_3.mp4")
        elif self.video_background == "video_4":
            bg_path = os.path.join(backgrounds_dir, "video_4.mp4")
        else:
            logger.warning(f"Unknown background: {self.video_background}. Using video_1.")
            bg_path = os.path.join(backgrounds_dir, "video_1.mp4")
        
        # For video files, check if they exist
        if self.video_background != "audio_particles" and os.path.exists(bg_path):
            logger.info(f"Using pre-rendered background: {bg_path}")
            return bg_path
        else:
            logger.error(f"Background file not found: {bg_path}. Using black background.")
            return None

    def transcribe(self, audio_filepath, artist=None, title=None, progress_callback=None):
        if self.demo_mode:
            return self._demo_transcribe(audio_filepath, artist, title, progress_callback)

        transcription_result = None
        correction_result = None
        self.corrected_lyrics = None
        segments_to_process = []
        resized_segments = []
        output_artifacts = None
        ass_file = None
        video_path_to_return = None
        lyrics_text = "Transcription failed or produced no text."
        download_files = []

        try:
            if progress_callback: progress_callback(0.1, "Initializing...")

            if not audio_filepath or not os.path.exists(audio_filepath):
                raise ValueError(f"Audio file path invalid or file not found: {audio_filepath}")

            if artist and title:
                output_prefix = f"{artist} - {title}"
            elif title:
                output_prefix = title
            else:
                output_prefix = Path(audio_filepath).stem

            logger.info(f"Using output prefix: {output_prefix}")

            audio_for_video = audio_filepath
            if self.enable_separate_vocals:
                if progress_callback: progress_callback(0.15, "Separating vocals...")
                try:
                    audio_for_video = self.separate_vocals(audio_filepath, output_prefix)
                    if not os.path.exists(audio_for_video):
                        logger.error(f"Instrumental audio not found after separation: {audio_for_video}")
                        audio_for_video = audio_filepath
                        logger.warning("Falling back to original audio due to missing instrumental file.")
                    else:
                        logger.info(f"Using instrumental audio for video: {audio_for_video}")
                except Exception as e:
                    logger.error(f"Vocal separation failed: {str(e)}. Falling back to original audio.", exc_info=True)
                    audio_for_video = audio_filepath

            if self.use_beat_effects:
                if progress_callback: progress_callback(0.18, "Detecting beats...")
                beat_times, tempo = self.detect_beats(audio_for_video)
                logger.info(f"Beat detection complete. {len(beat_times)} beats detected.")
            else:
                beat_times = None
                logger.info("Beat effects disabled. Skipping beat detection.")

            if progress_callback: progress_callback(0.2, "Starting transcription...")
            transcription_result = self.transcriber.transcribe(audio_filepath)
            if not transcription_result:
                raise ValueError("Transcription failed or returned empty result")
            logger.info("Whisper transcription completed.")

            logger.debug(f"Transcription result type: {type(transcription_result)}")
            logger.debug(f"Transcription result: {transcription_result.to_dict() if isinstance(transcription_result, TranscriptionData) else transcription_result}")

            normalized_result = transcription_result
            if isinstance(transcription_result, TranscriptionData):
                normalized_result = transcription_result.to_dict()
            elif not isinstance(transcription_result, dict):
                raise ValueError(f"Unexpected transcription result type: {type(transcription_result)}")

            segments = normalized_result.get('segments', [])
            words = [Word(**word_dict) for word_dict in normalized_result.get('words', [])]
            text = normalized_result.get('text', '')
            metadata = normalized_result.get('metadata', {})

            segments = [
                seg for seg in (self._dict_to_lyrics_segment(seg_dict, words) for seg_dict in segments)
                if seg is not None
            ]

            logger.debug(f"Transcription segments: {segments}")
            logger.debug(f"Number of segments: {len(segments)}")
            logger.debug(f"Transcription words: {words}")
            logger.debug(f"Number of words: {len(words)}")
            logger.debug(f"Transcription text: {text}")

            if not segments:
                logger.error("No segments available after processing. Cannot proceed with processing.")
                raise ValueError("Transcription produced no segments. Ensure the audio contains detectable speech.")

            logger.debug(f"Normalized transcription result: {normalized_result}")

            del self.transcriber
            self.transcriber = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            if self.use_llm_correction and self.corrector:
                if progress_callback: progress_callback(0.6, "Processing lyrics (Correction)...")
                try:
                    transcription_for_correction = transcription_result
                    if isinstance(transcription_result, dict):
                        transcription_for_correction = TranscriptionData(
                            segments=[LyricsSegment(**seg) for seg in transcription_result.get('segments', [])],
                            words=[Word(**word) for word in transcription_result.get('words', [])],
                            text=transcription_result.get('text', ''),
                            metadata=transcription_result.get('metadata', {})
                        )
                    transcription_results_list = [TranscriptionResult(
                        name="whisper",
                        priority=1,
                        result=transcription_for_correction
                    )]
                    correction_result = self.corrector.run(
                        transcription_results=transcription_results_list,
                        lyrics_results={},
                        metadata={"audio_file_hash": None}
                    )
                    self.corrected_lyrics = correction_result
                    if self.corrected_lyrics and hasattr(self.corrected_lyrics, 'corrected_segments'):
                        segments_to_process = self.corrected_lyrics.corrected_segments
                        logger.info(f"Correction successful. Got {len(segments_to_process)} segments.")
                    else:
                        logger.warning("Correction result did not contain 'corrected_segments'. Falling back.")
                        segments_to_process = [
                            seg for seg in segments
                            if self._get_segment_start_time(seg) >= self.post_intro_skip_seconds
                        ]
                        logger.warning(f"Using segments from original transcription (post-intro skip of {self.post_intro_skip_seconds} seconds).")
                except Exception as e:
                    logger.error(f"Correction step failed: {e}", exc_info=True)
                    logger.warning("Falling back to using original transcription segments (if available).")
                    segments_to_process = [
                        seg for seg in segments
                        if self._get_segment_start_time(seg) >= self.post_intro_skip_seconds
                    ]
                    self.corrected_lyrics = None
                finally:
                    if self.corrector:
                        for handler in self.corrector.handlers:
                            if hasattr(handler, 'provider') and hasattr(handler.provider, 'unload_model'):
                                handler.provider.unload_model()
                        del self.corrector
                        self.corrector = None
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                logger.info("Skipping LLM correction step. Using Whisper output directly.")
                segments_to_process = [
                    seg for seg in segments
                    if self._get_segment_start_time(seg) >= self.post_intro_skip_seconds
                ]
                self.corrected_lyrics = None

            resized_segments = []
            try:
                if segments_to_process:
                    logger.info("Attempting segment resizing...")
                    resized_segments = self.segment_resizer.resize_segments(segments_to_process)
                    logger.info(f"Resized segments count: {len(resized_segments)}")
                    for idx, seg in enumerate(resized_segments):
                        logger.debug(f"Resized segment {idx}: text='{seg.text}', time={seg.start_time}-{seg.end_time}")
                else:
                    logger.warning("Skipping segment resizing as there are no input segments.")
            except Exception as resize_e:
                logger.error(f"Error during segment resizing: {resize_e}", exc_info=True)
                resized_segments = segments_to_process
                logger.warning("Using un-resized segments due to resizing error.")

            if progress_callback: progress_callback(0.8, "Generating output files...")
            output_artifacts = self.output_generator.generate_outputs(
                transcription_corrected=self.corrected_lyrics,
                lyrics_results={},
                output_prefix=output_prefix,
                audio_filepath=audio_filepath,
                artist=artist,
                title=title
            )

            try:
                segments_for_ass = resized_segments
                if segments_for_ass:
                    logger.info("Generating ASS file...")
                    ass_file = self.subtitle_generator.generate_ass(
                        segments=segments_for_ass,
                        output_prefix=output_prefix,
                        audio_filepath=audio_filepath
                    )
                    if ass_file and os.path.exists(ass_file):
                        logger.info(f"ASS file generated: {ass_file}")
                        with open(ass_file, 'r', encoding='utf-8') as f:
                            ass_content = f.readlines()
                            logger.debug("ASS file content (first 20 lines):")
                            for line in ass_content[:20]:
                                try:
                                    logger.debug(line.strip())
                                except UnicodeEncodeError:
                                    logger.debug(f"Line with encoding issue: {repr(line.strip())}")
                        output_artifacts.ass = ass_file
                    else:
                        logger.error(f"Subtitle generator returned path '{ass_file}', but file not found.")
                        ass_file = None
                else:
                    logger.warning("No segments available to generate ASS file.")
            except Exception as e:
                logger.error(f"Failed to generate ASS file: {e}", exc_info=True)
                ass_file = None

            if ass_file:
                logger.info(f"Rendering video with ASS file: {ass_file}")
                video_suffix = "Karaoke" if self.enable_separate_vocals and audio_for_video != audio_filepath else "With Vocals"
                video_render_output_path = os.path.join(self.output_dir, f"{output_prefix} ({video_suffix}).mp4")
                try:
                    current_background_image = self.background_image
                    if self.video_background == "audio_particles":
                        current_background_image = "audio_particles"
                    elif self.video_background:
                        background_path = os.path.join(
                            os.path.dirname(__file__),
                            "backgrounds",
                            f"{self.video_background}.mp4"
                        )
                        if not os.path.exists(background_path):
                            logger.warning(f"Background video not found: {background_path}. Using black.")
                            background_path = None
                        current_background_image = background_path
                    else:
                        current_background_image = None

                        if not os.path.exists(background_path):
                            logger.warning(f"Background video not found: {background_path}. Using black.")
                            background_path = None

                    if not current_background_image or (current_background_image != "audio_particles" and not os.path.exists(current_background_image)):
                        logger.warning(f"Background image not found: {current_background_image}. Using black.")
                        current_background_image = None if current_background_image != "audio_particles" else "audio_particles"

                    logger.debug(f"Rendering video with audio: {audio_for_video}")
                    rendered_video_path = render_video_with_background(
                        audio_filepath=audio_for_video,
                        ass_filepath=ass_file,
                        output_filepath=video_render_output_path,
                        background_image=current_background_image,
                        resolution=self.resolution,
                        beat_times=self.beat_times,
                    )

                    if rendered_video_path and os.path.exists(rendered_video_path):
                        video_path_to_return = rendered_video_path
                        output_artifacts.video = video_path_to_return
                        logger.info(f"Video rendered: {video_path_to_return}")
                    else:
                        logger.error(f"Video rendering failed, returned '{rendered_video_path}'.")
                        video_path_to_return = None
                except Exception as render_e:
                    logger.error(f"Video rendering error: {render_e}", exc_info=True)
                    video_path_to_return = None
            else:
                logger.warning("Skipping video rendering due to missing ASS file.")

            if progress_callback: progress_callback(1.0, "Processing complete!")

            download_files = []
            try:
                for attr in ['lrc', 'ass', 'video', 'original_txt', 'corrected_txt', 'corrections_json']:
                    path = getattr(output_artifacts, attr, None)
                    if path and os.path.exists(path):
                        download_files.append(path)
                logger.info(f"Prepared download files: {download_files}")
            except Exception as e:
                logger.error(f"Error collecting download files: {e}")

            try:
                lyrics_text = self._format_lyrics_from_segments(resized_segments)
            except Exception as fmt_e:
                logger.error(f"Error formatting lyrics: {fmt_e}")
                lyrics_text = "Error formatting lyrics."

            logger.info(f"Transcription successful. Video path: {video_path_to_return}")

            return {
                "success": True if video_path_to_return else False,
                "message": "Transcription completed." + (" Video generated." if video_path_to_return else " Video generation failed."),
                "lyrics_text": lyrics_text,
                "download_files": download_files,
                "results": transcription_result,
                "video_path": video_path_to_return,
                "ass": ass_file,
                "video": video_path_to_return,
                "lrc": output_artifacts.lrc,
                "original_txt": output_artifacts.original_txt,
                "corrected_txt": output_artifacts.corrected_txt,
                "corrections_json": output_artifacts.corrections_json
            }

        except Exception as e:
            logger.error(f"Error in transcription process: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Error during processing: {str(e)}",
                "lyrics_text": None,
                "download_files": [],
                "results": transcription_result,
                "video_path": None
            }

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
        base_font_size = 48
        base_line_height = 60
        base_height = 720

        if resolution not in resolution_map:
            logger.warning(f"Invalid resolution '{resolution}'. Defaulting to 1080p.")
            resolution = "1080p"

        dims = resolution_map[resolution]
        height = dims[1]
        scaling_factor = height / base_height

        if resolution == "360p":
            scaling_factor = 0.5
        elif resolution == "720p":
            scaling_factor = 1.0
        elif resolution == "1080p":
            scaling_factor = 1.8
        elif resolution == "4k":
            scaling_factor = 3.5

        font_size = int(base_font_size * scaling_factor)
        line_height = int(base_line_height * scaling_factor)
        max_font_size = 200
        max_line_height = 200
        font_size = min(font_size, max_font_size)
        line_height = min(line_height, max_line_height)

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