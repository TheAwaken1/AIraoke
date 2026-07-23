
"""
Main application file for the Lyrics Transcriber Gradio UI
"""

import gradio as gr
import os
import sys
from pathlib import Path
import logging
import asyncio
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from fastapi import FastAPI, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

def smart_validate_file(file_path, allow_video=False):
    """
    Validates file extension. If allow_video=True, also accepts common video formats.
    """
    if not file_path or not os.path.exists(file_path):
        return False, "File not found."
    
    valid_audio = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.mp4', '.mkv', '.webm', '.mov'}
    valid_video = {'.mp4', '.mkv', '.webm', '.mov', '.avi'}
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if allow_video:
        if ext in valid_audio or ext in valid_video:
            return True, "Valid file."
        else:
            return False, "Unsupported file format. Supported: audio files + .mp4, .mkv, .webm, .mov"
    else:
        if ext in valid_audio:
            return True, "Valid audio file."
        else:
            return False, "Unsupported audio format. Please upload .mp3, .wav, .m4a, .flac, .ogg, .aac"

# Configure logging
log_level = os.getenv("GRADIO_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "gradio_ui.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("gradio_ui").setLevel(logging.INFO)
logging.getLogger("gradio_ui.transcriber").setLevel(logging.INFO)
logging.getLogger("gradio_ui.video_renderer").setLevel(logging.INFO)
logging.getLogger("python_multipart").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# File handler for DEBUG logs
# file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), "gradio_ui.log"), encoding='utf-8')
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
# logging.getLogger("gradio_ui.transcriber").addHandler(file_handler)
# logging.getLogger("gradio_ui.video_renderer").addHandler(file_handler)

# Unset Whisper environment variable
os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)

app_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(app_parent_dir)

from gradio_ui.transcriber import LyricsTranscriberWrapper
from gradio_ui.enhanced_features import LyricsDisplayManager
from gradio_ui.utils import validate_audio_file
from gradio_ui.gpu_utils import check_cuda_availability, configure_torch_for_gpu
from gradio_ui.song_history import get_history_manager
from lyrics_transcriber.types import LyricsSegment, Word

# Middleware to handle large responses
class ContentLengthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        logger.debug(f"Request path: {request.url.path}, Headers before: {response.headers}")
        response.headers["Transfer-Encoding"] = "chunked"
        response.headers.pop("Content-Length", None)
        logger.debug(f"Headers after: {response.headers}")
        return response

# Middleware to limit file upload size
class FileSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_size: int):
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds limit of {self.max_size // (1024 * 1024)} MB"
            )
        return await call_next(request)

lyrics_display_manager = LyricsDisplayManager()

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#111827",
    block_background_fill="#1f2937",
    block_border_width="1px",
    block_shadow="*shadow_drop_lg",
    block_label_background_fill="*neutral_800",
    block_label_text_color="white",
    block_title_text_color="white",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
    button_secondary_background_fill="*neutral_700",
    button_secondary_background_fill_hover="*neutral_600",
    button_secondary_text_color="white",
    input_background_fill="#374151",
    input_border_color="*neutral_700",
    input_placeholder_color="*neutral_500",
    link_text_color="*primary_400",
    link_text_color_hover="*primary_300",
    background_fill_primary="#111827",
    background_fill_secondary="#1f2937",
    border_color_accent="*primary_500",
    border_color_primary="*neutral_700",
    color_accent="*primary_500",
    color_accent_soft="*primary_800",
    shadow_spread="3px",
    shadow_drop="rgba(0,0,0,0.2)",
    shadow_drop_lg="rgba(0,0,0,0.3)",
)

# 1. Update VIDEO_BACKGROUNDS dictionary (remove PNG Slideshow)
VIDEO_BACKGROUNDS = {
    "Video 1": "video_1",
    "Video 2": "video_2",
    "Video 3": "video_3",
    "Video 4": "video_4",
    "Custom Video": "custom",  # user-uploaded background video
    "Audio Particles": "audio_particles",
    "Black": "black"
}

VIDEO_RESOLUTIONS = {
    "360p": "360p",
    "720p": "720p",
    "1080p": "1080p",
    "4k": "4k"
}

QUALITY_PRESETS = {
    "Fast": {"model_size": "turbo", "resolution": "720p"},
    "Balanced": {"model_size": "large-v3", "resolution": "1080p"},
    "Best": {"model_size": "large-v3", "resolution": "4k"},
}

FONT_COLORS = {
    "Fire Red": "255,69,0,255",
    "Neon Green": "57,255,20,255",
    "Electric Blue": "0,191,255,255",
    "Lime": "0,255,0,255",
    "Orange": "255,165,0,255",
    "Violet": "238,130,238,255",
    "Pink": "255,105,180,255",
    "Gold": "255,215,0,255",
    "Neon Purple": "191,0,255,255",
}

custom_css = """
/* ===== STEP PROGRESS INDICATOR ===== */
.step-indicator {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0;
    padding: 1rem 2rem;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, rgba(31,41,55,0.9) 0%, rgba(17,24,39,0.95) 100%);
    border-radius: 12px;
    border: 1px solid #374151;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.25rem;
    border-radius: 8px;
    transition: all 0.3s ease;
    background: transparent;
}
.step-item.active {
    background: linear-gradient(135deg, rgba(139,92,246,0.3) 0%, rgba(59,130,246,0.3) 100%);
    border: 1px solid rgba(139,92,246,0.5);
    box-shadow: 0 0 20px rgba(139,92,246,0.3);
}
.step-item.completed {
    background: rgba(34,197,94,0.2);
    border: 1px solid rgba(34,197,94,0.4);
}
.step-number {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 0.9rem;
    background: #374151;
    color: #9ca3af;
    border: 2px solid #4b5563;
}
.step-item.active .step-number {
    background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);
    color: white;
    border-color: transparent;
    box-shadow: 0 0 10px rgba(139,92,246,0.5);
}
.step-item.completed .step-number {
    background: #22c55e;
    color: white;
    border-color: transparent;
}
.step-label {
    font-size: 0.9rem;
    color: #9ca3af;
    font-weight: 500;
}
.step-item.active .step-label {
    color: white;
    font-weight: 600;
}
.step-item.completed .step-label {
    color: #86efac;
}
.step-arrow {
    color: #4b5563;
    font-size: 1.2rem;
    margin: 0 0.25rem;
}
.step-item.completed + .step-arrow {
    color: #22c55e;
}

/* ===== SECTION STYLES ===== */
.section-group { padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); }
.section-accordion { margin-bottom: 1rem; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); overflow: hidden; }
.section-accordion > :first-child { padding: 1rem 1.5rem; border-radius: 8px 8px 0 0; margin-bottom: 0; border: none; }
.section-accordion > :last-child { border-radius: 0 0 8px 8px; border: 1px solid #374151; border-top: none; padding: 1.5rem; background-color: #1f2937; }
.section-outputs { padding: 1.5rem; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); }
.section-upload { background: linear-gradient(90deg, rgba(139,92,246,1) 0%, rgba(59,130,246,1) 100%); }
.section-settings > :first-child { background: linear-gradient(90deg, rgba(139,92,246,1) 0%, rgba(59,130,246,1) 100%); }
.section-outputs { background: linear-gradient(90deg, rgba(59,130,246,1) 0%, rgba(139,92,246,1) 100%); }
.section-upload label, .section-upload button, .section-upload div, .section-upload span, .section-upload input::placeholder, .section-upload textarea::placeholder { color: white !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); }
.section-settings > :first-child span { color: white !important; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); }
.section-settings > :last-child label, .section-settings > :last-child span { color: #d1d5db; }
.section-outputs .tab-nav button, .section-outputs label, .section-outputs button:not(.primary), .section-outputs div, .section-outputs span, .section-outputs textarea::placeholder { color: white !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); }
.section-outputs .gradio-video label span { color: white !important; }
.section-outputs .gradio-html { color: white !important; }
.section-outputs .tab-nav button { background-color: transparent !important; border: none !important; }
.section-outputs .tab-nav button.selected { border-bottom: 2px solid white !important; }

/* ===== SETTINGS CATEGORY STYLES ===== */
.settings-category {
    margin-bottom: 0.75rem;
    border: 1px solid #374151;
    border-radius: 8px;
    overflow: hidden;
}
.settings-category-header {
    background: linear-gradient(90deg, rgba(75,85,99,0.5) 0%, rgba(55,65,81,0.5) 100%);
    padding: 0.5rem 1rem;
    font-weight: 600;
    color: #e5e7eb;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.settings-category-header::before {
    font-family: "Font Awesome 6 Free";
    font-weight: 900;
}
.settings-audio .settings-category-header::before { content: "🎵"; }
.settings-video .settings-category-header::before { content: "🎬"; }
.settings-ai .settings-category-header::before { content: "🤖"; }
.settings-effects .settings-category-header::before { content: "✨"; }

/* ===== STATUS BOX STYLING ===== */
#status-box textarea {
    font-size: 1rem !important;
    font-weight: 500 !important;
    padding: 0.75rem !important;
    border-radius: 8px !important;
}
#status-box.success textarea {
    background: rgba(34, 197, 94, 0.15) !important;
    border-color: rgba(34, 197, 94, 0.5) !important;
    color: #86efac !important;
}
#status-box.error textarea {
    background: rgba(239, 68, 68, 0.15) !important;
    border-color: rgba(239, 68, 68, 0.5) !important;
    color: #fca5a5 !important;
}

/* ===== BUTTON IMPROVEMENTS ===== */
button.primary.lg {
    font-size: 1.1rem !important;
    padding: 1rem 2rem !important;
    min-width: 200px !important;
}

/* ===== EDITOR HINT ===== */
#editor-hint {
    color: #9ca3af;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
}

/* ===== NEXT STEP HINT BOX ===== */
.next-step-hint {
    background: linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(139,92,246,0.15) 100%);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
}
.next-step-hint .hint-icon {
    font-size: 1.5rem;
    line-height: 1;
    flex-shrink: 0;
}
.next-step-hint .hint-content {
    flex: 1;
}
.next-step-hint .hint-title {
    font-weight: 600;
    color: #a5b4fc;
    font-size: 0.9rem;
    margin-bottom: 0.25rem;
}
.next-step-hint .hint-message {
    color: #e5e7eb;
    font-size: 0.95rem;
    line-height: 1.4;
}
.next-step-hint .hint-message strong {
    color: #fbbf24;
}
.next-step-hint.success {
    background: linear-gradient(135deg, rgba(34,197,94,0.15) 0%, rgba(16,185,129,0.15) 100%);
    border-color: rgba(34,197,94,0.4);
}
.next-step-hint.success .hint-title {
    color: #86efac;
}
.next-step-hint.ready {
    background: linear-gradient(135deg, rgba(251,191,36,0.15) 0%, rgba(245,158,11,0.15) 100%);
    border-color: rgba(251,191,36,0.4);
    animation: pulse-hint 2s ease-in-out infinite;
}
.next-step-hint.ready .hint-title {
    color: #fcd34d;
}
@keyframes pulse-hint {
    0%, 100% { box-shadow: 0 0 0 0 rgba(251,191,36,0.2); }
    50% { box-shadow: 0 0 15px 2px rgba(251,191,36,0.3); }
}

/* ===== VISUAL HIGHLIGHT FOR SECTIONS ===== */
.section-highlight {
    animation: section-pulse 2s ease-in-out infinite;
    position: relative;
}
.section-highlight::after {
    content: '';
    position: absolute;
    inset: -2px;
    border-radius: 10px;
    border: 2px solid rgba(251,191,36,0.6);
    pointer-events: none;
    animation: border-pulse 2s ease-in-out infinite;
}
@keyframes section-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(251,191,36,0.1); }
    50% { box-shadow: 0 0 20px 5px rgba(251,191,36,0.2); }
}
@keyframes border-pulse {
    0%, 100% { border-color: rgba(251,191,36,0.3); }
    50% { border-color: rgba(251,191,36,0.7); }
}

/* ===== BUTTON STATES ===== */
button.primary.lg:disabled,
button.secondary:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
    background: #4b5563 !important;
}
button.primary.lg:disabled:hover,
button.secondary:disabled:hover {
    transform: none !important;
    box-shadow: none !important;
}

/* Button ready state - pulse when action available */
.btn-ready {
    animation: btn-pulse 2s ease-in-out infinite;
}
@keyframes btn-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(139,92,246,0.4); }
    50% { box-shadow: 0 0 15px 3px rgba(139,92,246,0.6); }
}

/* ===== ENHANCED STEP INDICATOR WITH SUB-STEPS ===== */
.step-item .sub-steps {
    display: flex;
    gap: 0.25rem;
    margin-top: 0.25rem;
    justify-content: center;
}
.step-item .sub-step {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #4b5563;
    transition: all 0.3s ease;
}
.step-item .sub-step.done {
    background: #22c55e;
}
.step-item .sub-step.active {
    background: #8b5cf6;
    box-shadow: 0 0 6px rgba(139,92,246,0.6);
}
.step-item.completed .sub-step {
    background: #22c55e;
}

/* ===== BUTTON HINT TEXT ===== */
#transcribe-btn-hint, #render-btn-hint {
    text-align: center;
    margin-top: 0.25rem;
    font-size: 0.8rem;
}
#transcribe-btn-hint p, #render-btn-hint p {
    color: #9ca3af !important;
    margin: 0;
}
#app-title h1 {
    color: white !important;
    background: none !important;
    padding-bottom: 5px;
    display: inline-block;
}
/* Custom highlight bar style */
.lyric-line::before {
    content: '';
    display: inline-block;
    width: 10px;
    height: 20px;
    margin-right: 10px;
    background: linear-gradient(90deg, #FFD700, #FF69B4);
    box-shadow: 0 0 10px rgba(255, 215, 0, 0.8);
    border-radius: 5px;
    animation: pulse 1.5s infinite ease-in-out, gradientShift 3s infinite ease-in-out;
}
.lyric-line.active::before {
    background: linear-gradient(90deg, #00FFFF, #FF00FF);
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.9);
    animation: pulse 1s infinite ease-in-out, gradientShiftActive 2s infinite ease-in-out;
}
@keyframes pulse {
    0% { box-shadow: 0 0 10px rgba(255, 215, 0, 0.8); }
    50% { box-shadow: 0 0 20px rgba(255, 215, 0, 1); }
    100% { box-shadow: 0 0 10px rgba(255, 215, 0, 0.8); }
}
@keyframes gradientShift {
    0% { background: linear-gradient(90deg, #FFD700, #FF69B4); }
    50% { background: linear-gradient(90deg, #FF69B4, #FFD700); }
    100% { background: linear-gradient(90deg, #FFD700, #FF69B4); }
}
@keyframes pulseActive {
    0% { box-shadow: 0 0 15px rgba(0, 255, 255, 0.9); }
    50% { box-shadow: 0 0 25px rgba(0, 255, 255, 1); }
    100% { box-shadow: 0 0 15px rgba(0, 255, 255, 0.9); }
}
@keyframes gradientShiftActive {
    0% { background: linear-gradient(90deg, #00FFFF, #FF00FF); }
    50% { background: linear-gradient(90deg, #FF00FF, #00FFFF); }
    100% { background: linear-gradient(90deg, #00FFFF, #FF00FF); }
}
"""

import re

def parse_artist_title_from_filename(media_path):
    """Best-effort 'Artist - Title' extraction from a media filename."""
    if not media_path:
        return "", ""
    name = os.path.splitext(os.path.basename(media_path))[0]
    # Strip bracketed noise like (Official Video), [Lyrics], (Official Audio)
    name = re.sub(r"[\(\[][^)\]]*(official|video|audio|lyric)[^)\]]*[\)\]]", "", name, flags=re.I)
    name = name.strip(" -_")
    for sep in [" - ", " – ", "- "]:
        if sep in name:
            artist, title = name.split(sep, 1)
            return artist.strip(), title.strip()
    return "", name.strip()


def validate_audio_content(audio_filepath):
    """Validate the audio file for duration and detectable speech."""
    try:
        audio = AudioSegment.from_file(audio_filepath)
        duration_ms = len(audio)
        if duration_ms < 5000:
            return False, "Audio file is too short (less than 5 seconds)."
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)
        if not nonsilent_ranges:
            return False, "Audio file contains no detectable speech or vocals (mostly silent)."
        nonsilent_duration_ms = sum(end - start for start, end in nonsilent_ranges)
        nonsilent_percentage = (nonsilent_duration_ms / duration_ms) * 100
        if nonsilent_percentage < 10:
            return False, "Audio file contains minimal detectable speech or vocals (less than 10% non-silent content)."
        return True, "Audio validation successful."
    except Exception as e:
        logger.error(f"Error validating audio content: {e}")
        return False, f"Error validating audio content: {str(e)}"


def apply_time_offset(segments, offset: float):
    """Apply a time offset to all segments and their words."""
    from lyrics_transcriber.types import LyricsSegment, Word

    if offset == 0:
        return segments

    adjusted_segments = []
    for seg in segments:
        new_start = max(0, seg.start_time + offset)
        new_end = max(0, seg.end_time + offset)

        adjusted_words = []
        for word in seg.words:
            adjusted_words.append(Word(
                id=word.id,
                text=word.text,
                start_time=max(0, word.start_time + offset),
                end_time=max(0, word.end_time + offset),
                confidence=word.confidence,
                created_during_correction=getattr(word, 'created_during_correction', False)
            ))

        adjusted_segments.append(LyricsSegment(
            id=seg.id,
            text=seg.text,
            words=adjusted_words,
            start_time=new_start,
            end_time=new_end
        ))

    return adjusted_segments


def transcribe_audio_only(
    audio_file, artist_name, song_title,
    use_gpu, model_size,
    use_llm_correction,
    llm_corrector_model,
    enable_beat_effects,
    use_input_video,
    lyrics_time_offset,
    progress=gr.Progress()
):
    # Keep step indicator at step 1 (Upload) on errors
    step_html_error = generate_step_indicator(current_step=1, completed_steps=[])
    hint_html_error = generate_next_step_hint(has_audio=False)
    render_btn_disabled = gr.update(interactive=False)
    render_hint_disabled = gr.update(value="*Transcribe lyrics first to enable*")

    if audio_file is None:
        logger.warning("No audio file uploaded.")
        return "⚠️ Please upload an audio file first.", None, None, None, [], [], gr.update(), step_html_error, hint_html_error, render_btn_disabled, render_hint_disabled, gr.update(), gr.update(), gr.update()
    is_valid, message = smart_validate_file(audio_file, allow_video=use_input_video)
    if not is_valid:
        logger.warning(f"Invalid audio file: {message}")
        return f"⚠️ {message}", None, None, None, [], [], gr.update(), step_html_error, hint_html_error, render_btn_disabled, render_hint_disabled, gr.update(), gr.update(), gr.update()

    is_valid_content, content_message = validate_audio_content(audio_file)
    if not is_valid_content:
        logger.warning(f"Audio content validation failed: {content_message}")
        return f"⚠️ {content_message}", None, None, None, [], [], gr.update(), step_html_error, hint_html_error, render_btn_disabled, render_hint_disabled, gr.update(), gr.update(), gr.update()

    try:
        app_dir = os.path.dirname(__file__)
        output_dir = os.path.join(app_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        return f"⚠️ Error creating output directory: {e}", None, None, None, [], [], gr.update(), step_html_error, hint_html_error, render_btn_disabled, render_hint_disabled, gr.update(), gr.update(), gr.update()

    # Normal Whisper transcription flow
    try:
        # Initialize Wrapper for Transcription
        transcriber = LyricsTranscriberWrapper(
            use_gpu=use_gpu,
            output_dir=output_dir,
            model_size=model_size,
            # Video settings are not needed for this phase but required by init
            video_background="video_1",
            font_color="255,255,255,255",
            resolution="1080p",
            progress_callback=progress,
            use_llm_correction=use_llm_correction,
            llm_corrector_model=llm_corrector_model,
            enable_separate_vocals=False,  # vocal separation is handled at render time via vocal volume
            use_beat_effects=enable_beat_effects,
            # video_effect and use_input_video handled in rendering,
            # but input video file check might happen here?
            video_effect="None",
            use_input_video=use_input_video,
        )
    except Exception as e:
        hint_html_err = generate_next_step_hint(has_audio=True)
        return f"⚠️ Error initializing transcriber: {e}", None, None, None, [], [], gr.update(), step_html_error, hint_html_err, render_btn_disabled, render_hint_disabled, gr.update(), gr.update(), gr.update()

    def update_progress(value, description=None):
        if description:
            progress(value, desc=description)
        else:
            progress(value)

    try:
        result = transcriber.transcribe_audio(
            audio_file,
            progress_callback=update_progress,
        )
    except Exception as e:
        logger.error(f"Error during transcription audio process: {e}", exc_info=True)
        hint_html_err = generate_next_step_hint(has_audio=True)
        return f"⚠️ Error during transcription: {e}", None, None, None, [], [], gr.update(), step_html_error, hint_html_err, render_btn_disabled, render_hint_disabled, gr.update(), gr.update(), gr.update()

    if not result or not result.get("success"):
        message = result.get("message", "Unknown transcription error")
        hint_html_err = generate_next_step_hint(has_audio=True)
        return f"⚠️ {message}", None, None, None, [], [], gr.update(), step_html_error, hint_html_err, render_btn_disabled, render_hint_disabled, gr.update(), gr.update(), gr.update()

    segments = result.get("segments", [])
    beat_times = result.get("beat_times", [])
    lyrics_text = result.get("lyrics_text", "")
    status_msg = "Transcription complete (Whisper)"

    # Format for DataFrame: [Start, End, Text]
    df_data = [[seg.start_time, seg.end_time, seg.text] for seg in segments]

    try:
        styled_lyrics = lyrics_display_manager.format_lyrics_for_display(
            lyrics_text,
            show_upcoming=False
        )
    except Exception as e:
        styled_lyrics = "Error formatting lyrics display."

    # Add to song history (fall back to the filename when no artist/title was entered)
    try:
        artist_for_history = (artist_name or "").strip()
        title_for_history = (song_title or "").strip()
        if not artist_for_history and not title_for_history:
            artist_for_history, title_for_history = parse_artist_title_from_filename(audio_file)
        history_manager = get_history_manager()
        history_manager.add_song(
            artist=artist_for_history,
            title=title_for_history,
            audio_path=audio_file,
            lyrics_source="whisper",
            metadata={"model_size": model_size}
        )
    except Exception as e:
        logger.warning(f"Failed to save to history: {e}")

    # Generate step indicator showing step 3 (Edit) as active, steps 1-2 completed
    step_html = generate_step_indicator(
        current_step=3,
        completed_steps=[1, 2],
        sub_steps={3: (0, 2, 1)}  # Review is active
    )
    hint_html = generate_next_step_hint(has_audio=True, has_transcription=True, just_transcribed=True)

    return (
        f"✓ {status_msg}",
        lyrics_text,
        styled_lyrics,
        df_data,  # Dataframe
        segments,  # State
        beat_times,  # State
        gr.update(selected=1),  # Switch to Lyrics Editor tab (id=1)
        step_html,  # Update step indicator
        hint_html,  # Update hint
        gr.update(interactive=True, elem_classes=["btn-ready"]),  # Enable render button with pulse
        gr.update(value="*Review lyrics, pick your style, then click to create video*"),  # Update render hint
        gr.update(open=True),  # Open Video & Audio Style accordion
        gr.update(elem_classes=[]),  # Stop pulsing the transcribe button
        gr.update(choices=get_history_choices())  # Refresh Recent Songs dropdown
    )

def render_video_from_editor(
    editor_data, audio_file, artist_name, song_title,
    use_gpu, model_size, video_background, custom_background_video,
    font_color, resolution,
    enable_beat_effects, enable_countdown, enable_pitch_guide,
    video_effect, use_input_video, video_dimmer, vocal_volume,
    segments_state, beat_times_state,
    progress=gr.Progress()
):
    step_html_error = generate_step_indicator(current_step=3, completed_steps=[1, 2], sub_steps={3: (0, 2, 1)})
    hint_html_error = generate_next_step_hint(has_audio=True, has_transcription=True)
    render_hint_error = gr.update(value="*Review lyrics, then click to create video*")

    if not segments_state:
        return "⚠️ No transcription data. Please transcribe first (Step 2).", None, gr.update(), step_html_error, hint_html_error, render_hint_error, gr.update()

    if editor_data is None:
        return "⚠️ No lyrics data to render.", None, gr.update(), step_html_error, hint_html_error, render_hint_error, gr.update()

    if video_background == "Custom Video" and not custom_background_video:
        return "⚠️ Please upload a custom background video (or choose another background).", None, gr.update(), step_html_error, hint_html_error, render_hint_error, gr.update()

    # Reconstruct segments from editor data
    # editor_data is a list of lists: [[start, end, text], ...]
    updated_segments = []

    def create_words_from_text(text, start_time, end_time, original_words=None):
        """Create Word objects from text, distributing timing proportionally."""
        text = text.strip()
        if not text:
            return []

        # Split text into words
        word_texts = text.split()
        if not word_texts:
            return []

        duration = end_time - start_time
        words = []

        # If original words exist and word count matches, try to preserve timing ratios
        if original_words and len(original_words) == len(word_texts):
            # Calculate original total duration
            orig_duration = original_words[-1].end_time - original_words[0].start_time
            if orig_duration > 0:
                # Scale original word timings to new segment duration
                orig_start = original_words[0].start_time
                for idx, (word_text, orig_word) in enumerate(zip(word_texts, original_words)):
                    # Calculate relative position in original segment
                    rel_start = (orig_word.start_time - orig_start) / orig_duration
                    rel_end = (orig_word.end_time - orig_start) / orig_duration
                    # Apply to new segment timing
                    new_word_start = start_time + (rel_start * duration)
                    new_word_end = start_time + (rel_end * duration)
                    words.append(Word(
                        id=f"{idx}",
                        text=word_text,
                        start_time=new_word_start,
                        end_time=new_word_end,
                        confidence=1.0
                    ))
                return words

        # Otherwise, distribute timing proportionally based on character count
        total_chars = sum(len(w) for w in word_texts)
        if total_chars == 0:
            total_chars = len(word_texts)  # Fallback to equal distribution

        current_time = start_time
        for idx, word_text in enumerate(word_texts):
            # Calculate word duration based on character proportion
            word_duration = (len(word_text) / total_chars) * duration
            # Ensure minimum duration per word
            word_duration = max(word_duration, 0.1)
            word_end = min(current_time + word_duration, end_time)

            words.append(Word(
                id=f"{idx}",
                text=word_text,
                start_time=current_time,
                end_time=word_end,
                confidence=1.0
            ))
            current_time = word_end

        # Ensure last word ends at segment end
        if words:
            words[-1] = Word(
                id=words[-1].id,
                text=words[-1].text,
                start_time=words[-1].start_time,
                end_time=end_time,
                confidence=1.0
            )

        return words

    for i, row in enumerate(editor_data):
        try:
            if not row or len(row) < 3:
                continue

            start, end, text = row[:3]

            try:
                start = float(start) if start not in (None, "") else 0.0
                end = float(end) if end not in (None, "") else 0.0
            except ValueError:
                logger.warning(f"Invalid time format at row {i}: {start}, {end}")
                continue

            text = str(text).strip()

            original_seg = segments_state[i] if i < len(segments_state) else None

            if original_seg and original_seg.text == text:
                # Text matches exactly, keep original words but update segment timing
                # Scale word timings if segment timing changed
                if abs(original_seg.start_time - start) < 0.01 and abs(original_seg.end_time - end) < 0.01:
                    # Timing unchanged, use original words directly
                    words = original_seg.words
                else:
                    # Timing changed, scale word timings
                    words = create_words_from_text(text, start, end, original_seg.words)

                updated_segments.append(LyricsSegment(
                    id=original_seg.id,
                    text=text,
                    words=words,
                    start_time=float(start),
                    end_time=float(end)
                ))
            else:
                # Text changed or new segment - create words with distributed timing
                original_words = original_seg.words if original_seg else None
                words = create_words_from_text(text, start, end, original_words)

                updated_segments.append(LyricsSegment(
                    id=original_seg.id if original_seg else str(i),
                    text=text,
                    words=words,
                    start_time=float(start),
                    end_time=float(end)
                ))
        except Exception as e:
            logger.error(f"Error processing row {i}: {e}")
            continue

    try:
        app_dir = os.path.dirname(__file__)
        output_dir = os.path.join(app_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Ensure vocal_volume is an integer
        vocal_vol = int(vocal_volume) if vocal_volume is not None else 100
        logger.info(f"Vocal volume setting: {vocal_vol}% (will {'separate vocals' if vocal_vol < 100 else 'use original audio'})")

        transcriber = LyricsTranscriberWrapper(
            use_gpu=use_gpu,
            output_dir=output_dir,
            model_size=model_size,
            video_background=VIDEO_BACKGROUNDS.get(video_background, "video_1"),
            custom_background_path=custom_background_video,
            font_color=FONT_COLORS.get(font_color, "255,255,255,255"),
            resolution=VIDEO_RESOLUTIONS.get(resolution, "1080p"),
            progress_callback=progress,
            use_llm_correction=False, # Already corrected/edited
            llm_corrector_model=None,
            enable_separate_vocals=(vocal_vol < 100),  # Separate if we need to adjust vocals
            use_beat_effects=enable_beat_effects,
            video_effect=video_effect,
            use_input_video=use_input_video,
            video_dimmer=video_dimmer,
            enable_countdown=enable_countdown,
            enable_pitch_guide=enable_pitch_guide,
            vocal_volume=vocal_vol,
        )
    except Exception as e:
        return f"⚠️ Error initializing renderer: {e}", None, gr.update(), step_html_error, hint_html_error, render_hint_error, gr.update()

    def update_progress(value, description=None):
        if description:
            progress(value, desc=description)
        else:
            progress(value)

    result = transcriber.render_video(
        audio_file,
        updated_segments,
        beat_times_state,
        artist=artist_name,
        title=song_title,
        progress_callback=update_progress
    )
    
    if not result or not result.get("success"):
        # Keep step 3 active on error
        step_html = generate_step_indicator(current_step=3, completed_steps=[1, 2], sub_steps={3: (1, 2, 2)})
        hint_html = generate_next_step_hint(has_audio=True, has_transcription=True)
        return f"⚠️ {result.get('message', 'Render failed')}", None, gr.update(), step_html, hint_html, gr.update(value="*Fix issues and try again*"), gr.update()

    # Success - all steps completed, switch to video tab
    step_html = generate_step_indicator(
        current_step=4,
        completed_steps=[1, 2, 3, 4],
        sub_steps={4: (2, 2, 2)}  # All done
    )
    hint_html = generate_next_step_hint(has_video=True, just_rendered=True)
    return "✓ Video rendered successfully!", result.get("video_path"), gr.update(selected=3), step_html, hint_html, gr.update(value="*Video created! Re-render with different settings if needed*"), gr.update(elem_classes=[])

import yt_dlp

def download_youtube_video(url, download_video=False, progress=gr.Progress()):
    """Download video or audio from YouTube with robust error handling."""
    if not url or not url.strip():
        gr.Warning("Please enter a YouTube URL")
        return None

    progress(0, desc="Downloading from YouTube...")
    output_path = os.path.join(os.path.dirname(__file__), 'downloads')
    os.makedirs(output_path, exist_ok=True)

    # Common options for better compatibility with YouTube 2025/2026
    common_opts = {
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'ignoreerrors': False,
        'nocheckcertificate': True,
        'prefer_insecure': True,
        # Use cookies from browser if available
        'cookiesfrombrowser': ('chrome',),
        # Fallback: use android client which often works better
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
                'player_skip': ['webpage', 'configs'],
            }
        },
        # Retry settings
        'retries': 3,
        'fragment_retries': 3,
        'file_access_retries': 3,
    }

    if download_video:
        ydl_opts = {
            **common_opts,
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
        }
    else:
        ydl_opts = {
            **common_opts,
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }

    # Try without browser cookies first (faster), then with cookies as fallback
    for attempt, use_cookies in enumerate([(False, "without cookies"), (True, "with browser cookies")]):
        try:
            opts = ydl_opts.copy()
            if not use_cookies[0]:
                opts.pop('cookiesfrombrowser', None)

            progress(0.2, desc=f"Attempting download {use_cookies[1]}...")

            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    continue

                filename = ydl.prepare_filename(info)
                progress(0.9, desc="Processing...")

                if download_video:
                    base, ext = os.path.splitext(filename)
                    final_path = base + '.mp4' if ext != '.mp4' else filename
                    if os.path.exists(final_path):
                        progress(1.0, desc="Download complete!")
                        gr.Info(f"Downloaded: {os.path.basename(final_path)}")
                        return final_path
                    elif os.path.exists(filename):
                        progress(1.0, desc="Download complete!")
                        gr.Info(f"Downloaded: {os.path.basename(filename)}")
                        return filename
                else:
                    base, ext = os.path.splitext(filename)
                    wav_path = base + '.wav'
                    if os.path.exists(wav_path):
                        progress(1.0, desc="Download complete!")
                        gr.Info(f"Downloaded: {os.path.basename(wav_path)}")
                        return wav_path
                    # Try other audio formats as fallback
                    for audio_ext in ['.m4a', '.mp3', '.webm', '.opus']:
                        alt_path = base + audio_ext
                        if os.path.exists(alt_path):
                            progress(1.0, desc="Download complete!")
                            gr.Info(f"Downloaded: {os.path.basename(alt_path)}")
                            return alt_path

        except Exception as e:
            error_str = str(e)
            logger.warning(f"Download attempt {attempt + 1} failed: {error_str}")
            if attempt == 0:
                continue  # Try with cookies
            else:
                # Both attempts failed
                clean_error = error_str.replace('\x1b[0;31m', '').replace('\x1b[0m', '')
                logger.error(f"YouTube download failed: {clean_error}")
                gr.Error(f"Download failed: {clean_error[:200]}")
                return None

    gr.Error("Download failed after all attempts")
    return None

import subprocess

def get_ollama_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        models = [line.split()[0] for line in lines[1:] if line.strip()]  # Skip header/empty
        if not models:
            models = ["No models available (run 'ollama pull <model>')"]
        return models
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.warning(f"Error getting Ollama models: {e}. Ensure Ollama is installed.")
        return ["Ollama not available"]

def load_from_history(history_selection):
    """Load artist and title from history selection."""
    if not history_selection:
        return "", ""

    try:
        import json
        data = json.loads(history_selection)
        return data.get("artist", ""), data.get("title", "")
    except:
        return "", ""

def get_history_choices():
    """Get formatted history choices for dropdown."""
    history_manager = get_history_manager()
    choices = history_manager.format_for_dropdown()
    if not choices:
        return [("No history yet", "")]
    return choices

def generate_next_step_hint(
    has_audio: bool = False,
    has_transcription: bool = False,
    has_video: bool = False,
    just_transcribed: bool = False,
    just_rendered: bool = False
) -> str:
    """Generate contextual hint HTML based on current state."""

    if just_rendered or has_video:
        return '''
        <div class="next-step-hint success">
            <span class="hint-icon">🎉</span>
            <div class="hint-content">
                <div class="hint-title">Video Ready!</div>
                <div class="hint-message">Your karaoke video is complete! Check the <strong>Video Playback</strong> tab to watch or download it.</div>
            </div>
        </div>
        '''

    if just_transcribed or has_transcription:
        return '''
        <div class="next-step-hint success">
            <span class="hint-icon">✏️</span>
            <div class="hint-content">
                <div class="hint-title">Lyrics Ready - Review & Render</div>
                <div class="hint-message">Review the lyrics in the <strong>Editor tab</strong>, pick your <strong>Video & Audio Style</strong>, then click <strong>Render Video</strong>.</div>
            </div>
        </div>
        '''

    if has_audio:
        return '''
        <div class="next-step-hint ready">
            <span class="hint-icon">🎤</span>
            <div class="hint-content">
                <div class="hint-title">Media Loaded - Ready to Transcribe</div>
                <div class="hint-message">Click <strong>Transcribe Lyrics</strong> to let AI detect the words from your media.</div>
            </div>
        </div>
        '''

    # Initial state
    return '''
    <div class="next-step-hint">
        <span class="hint-icon">👋</span>
        <div class="hint-content">
            <div class="hint-title">Welcome to AIraoke!</div>
            <div class="hint-message">Start by <strong>uploading media</strong> or <strong>downloading from YouTube</strong>.</div>
        </div>
    </div>
    '''


def generate_step_indicator(
    current_step: int,
    completed_steps: list = None,
    sub_steps: dict = None
) -> str:
    """Generate HTML for step indicator with current step highlighted and sub-steps."""
    if completed_steps is None:
        completed_steps = []
    if sub_steps is None:
        sub_steps = {}

    # Define steps
    steps = [
        (1, "Upload"),
        (2, "Transcribe"),
        (3, "Edit"),
        (4, "Render")
    ]

    html_parts = ['<div class="step-indicator">']

    for i, (step_num, label) in enumerate(steps):
        classes = ["step-item"]
        if step_num in completed_steps:
            classes.append("completed")
        elif step_num == current_step:
            classes.append("active")

        html_parts.append(f'''
            <div class="{' '.join(classes)}" id="step-{step_num}">
                <span class="step-number">{step_num if step_num not in completed_steps else '✓'}</span>
                <span class="step-label">{label}</span>
            </div>
        ''')

        if i < len(steps) - 1:
            arrow_class = "step-arrow"
            html_parts.append(f'<span class="{arrow_class}">→</span>')

    html_parts.append('</div>')
    return ''.join(html_parts)

def create_ui():
    has_gpu, gpu_info, vram_gb = check_cuda_availability()
    combined_css = custom_css
    ollama_models = get_ollama_models()

    with gr.Blocks(theme=theme, title="AIraoke", css=combined_css) as app:
        app.app.add_middleware(ContentLengthMiddleware)
        app.app.add_middleware(FileSizeLimitMiddleware, max_size=100 * 1024 * 1024)
        
        # State variables
        segments_state = gr.State([])
        beat_times_state = gr.State([])

        gr.Markdown(
            """
            # AIraoke
            Transform your songs into karaoke videos with AI-powered lyrics transcription
            """,
            elem_id="app-title"
        )

        # Step Progress Indicator - use the function for consistency
        step_indicator = gr.HTML(
            value=generate_step_indicator(current_step=1, completed_steps=[], sub_steps={1: (0, 2, 1)}),
            elem_id="step-progress"
        )

        # Dynamic Next Step Hint
        next_step_hint = gr.HTML(
            value=generate_next_step_hint(),
            elem_id="next-step-hint"
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["section-group", "section-upload"]):
                    gr.Markdown("### 1. Upload Media")
                    with gr.Tabs() as upload_tabs:
                        with gr.Tab("Audio", id=0):
                            audio_file = gr.Audio(label="Upload Audio File", type="filepath", interactive=True)
                        with gr.Tab("Video", id=1):
                            video_file = gr.Video(label="Upload Video File", interactive=True)
                    
                    gr.Markdown("### OR")
                    youtube_url = gr.Textbox(label="YouTube URL", placeholder="Paste a YouTube link here")
                    with gr.Row():
                        download_video_checkbox = gr.Checkbox(label="Download Video (MP4)", value=False)
                        download_btn = gr.Button("Download from YouTube")

                    gr.Markdown("### 2. Song Info")
                    with gr.Row():
                        artist_name = gr.Textbox(label="Artist Name", placeholder="Enter artist name")
                        song_title = gr.Textbox(label="Song Title", placeholder="Enter song title")
                    
                    with gr.Row():
                        history_dropdown = gr.Dropdown(
                            label="Recent Songs",
                            choices=[],
                            interactive=True,
                            info="Load from history"
                        )
                        refresh_history_btn = gr.Button("↻", scale=0)
                    
                    # Manual offset moved to advanced or kept here? Kept here but simplified.
                    lyrics_time_offset = gr.Slider(
                        label="Manual Time Offset",
                        minimum=-30,
                        maximum=30,
                        value=0,
                        step=0.5,
                        visible=False # Hide by default as it's advanced
                    )

                quality_preset = gr.Radio(
                    label="⚡ Quality Preset",
                    choices=["Fast", "Balanced", "Best", "Custom"],
                    value="Balanced",
                    info="Fast: turbo model + 720p · Balanced: large-v3 + 1080p · Best: large-v3 + 4k"
                )

                with gr.Accordion("🎛️ Transcription Settings", open=False, elem_classes=["section-accordion", "section-settings"]) as transcription_settings_acc:
                    # Performance Settings
                    with gr.Accordion("🚀 Performance", open=True):
                        with gr.Row():
                            use_gpu = gr.Checkbox(
                                label=f"Use GPU {'✓ Available' if has_gpu else '✗ Not Available'}",
                                value=has_gpu,
                                interactive=has_gpu,
                                info="Enable CUDA acceleration for faster processing"
                            )
                            model_size = gr.Dropdown(
                                label="Whisper Model",
                                choices=["turbo", "large-v3"],
                                value="large-v3",
                                info="turbo = faster, large-v3 = more accurate"
                            )
                        if not has_gpu:
                            gr.Markdown(f"*{gpu_info}*")

                    # AI Enhancement
                    with gr.Accordion("🤖 AI Enhancement", open=False):
                        with gr.Row():
                            use_llm_correction = gr.Checkbox(
                                label="LLM Lyrics Correction",
                                value=False,
                                info="Use local AI to fix transcription errors"
                            )
                            llm_corrector_model = gr.Dropdown(
                                label="Ollama Model",
                                choices=ollama_models,
                                value=ollama_models[0] if ollama_models and ollama_models[0] != "No models available (run 'ollama pull <model>')" else None,
                                interactive=bool(ollama_models and ollama_models[0] != "No models available..."),
                                info="Select your local Ollama model",
                                visible=False
                            )

                    gr.Markdown(
                        "*🎵 Vocal volume, backgrounds, and effects are in **Video & Audio Style** inside the Lyrics Editor tab.*"
                    )

                transcribe_btn = gr.Button(
                    "🎤 Transcribe Lyrics",
                    variant="primary",
                    size="lg",
                    interactive=False,
                    elem_id="transcribe-btn"
                )
                transcribe_btn_hint = gr.Markdown(
                    "*Upload audio to enable*",
                    elem_id="transcribe-btn-hint",
                    visible=True
                )

            with gr.Column(scale=2, elem_classes=["section-outputs"]):
                with gr.Tabs(selected=0) as main_tabs:
                    with gr.TabItem("📝 Preview", id=0):
                        preview = gr.Textbox(
                            label="Lyrics Preview",
                            placeholder="Transcribed lyrics will appear here after Step 2...",
                            lines=20,
                            interactive=False
                        )
                    with gr.TabItem("✏️ Lyrics Editor", id=1):
                        gr.Markdown("### Edit Lyrics & Timing")
                        gr.Markdown("*Adjust start/end times and edit text. Changes will be used when rendering.*", elem_id="editor-hint")
                        editor_df = gr.Dataframe(
                            headers=["Start (sec)", "End (sec)", "Lyrics Text"],
                            datatype=["number", "number", "str"],
                            col_count=(3, "fixed"),
                            interactive=True,
                            label="Lyric Segments",
                            type="array"
                        )
                        with gr.Accordion("🎨 Video & Audio Style", open=False, elem_classes=["section-accordion", "section-settings"]) as video_style_acc:
                            with gr.Row():
                                video_background = gr.Dropdown(
                                    label="Background",
                                    choices=list(VIDEO_BACKGROUNDS.keys()),
                                    value="Video 1"
                                )
                                resolution = gr.Dropdown(
                                    label="Resolution",
                                    choices=list(VIDEO_RESOLUTIONS.keys()),
                                    value="1080p"
                                )
                            custom_background_video = gr.Video(
                                label="Custom Background Video (loops behind your lyrics)",
                                interactive=True,
                                visible=False
                            )
                            with gr.Row():
                                font_color = gr.Dropdown(
                                    label="Font Color",
                                    choices=list(FONT_COLORS.keys()),
                                    value="Fire Red"
                                )
                                video_effect = gr.Dropdown(
                                    label="Video Effect",
                                    choices=["None", "Black & White", "Sepia", "Vignette", "Blur", "Invert"],
                                    value="None"
                                )
                            with gr.Row():
                                use_input_video = gr.Checkbox(
                                    label="Use Uploaded Video as Background",
                                    value=False,
                                    info="Use your video file as the karaoke background",
                                    visible=False
                                )
                                video_dimmer = gr.Slider(
                                    label="Background Dimmer %",
                                    minimum=0,
                                    maximum=100,
                                    value=0,
                                    step=5,
                                    info="Darken background for better text visibility"
                                )
                            with gr.Row():
                                vocal_volume = gr.Slider(
                                    label="Vocal Volume %",
                                    minimum=0,
                                    maximum=100,
                                    value=100,
                                    step=5,
                                    info="0 = instrumental only. Below 100% enables vocal separation."
                                )
                            with gr.Row():
                                enable_beat_effects = gr.Checkbox(
                                    label="Beat Effects",
                                    value=True,
                                    info="Sync visual effects to music beats"
                                )
                                enable_countdown = gr.Checkbox(
                                    label="Countdown",
                                    value=True,
                                    info="Show 3-2-1-GO! before lyrics start"
                                )
                                enable_pitch_guide = gr.Checkbox(
                                    label="Pitch Guide (Beta)",
                                    value=False,
                                    info="Display pitch contour overlay"
                                )
                        with gr.Row():
                            render_btn = gr.Button(
                                "🎬 Render Video",
                                variant="primary",
                                size="lg",
                                interactive=False,
                                elem_id="render-btn"
                            )
                        render_btn_hint = gr.Markdown(
                            "*Transcribe lyrics first to enable*",
                            elem_id="render-btn-hint",
                            visible=True
                        )
                    with gr.TabItem("🎤 Lyrics Display", id=2):
                        lyrics_display = gr.HTML("Synchronized lyrics will appear here after transcription.")
                    with gr.TabItem("🎥 Video Playback", id=3):
                        video_output = gr.Video(label="Karaoke Video", interactive=False)

                error_output = gr.Textbox(label="Status", visible=True, interactive=False, elem_id="status-box")

        # Update step indicator, hint, and button states when media is uploaded
        def on_media_upload(audio_path, video_path, artist, title):
            media_path = audio_path or video_path
            if media_path:
                # Auto-fill artist/title from the filename, but never overwrite user input
                artist_update, title_update = gr.update(), gr.update()
                if not (artist or "").strip() and not (title or "").strip():
                    parsed_artist, parsed_title = parse_artist_title_from_filename(media_path)
                    if parsed_artist:
                        artist_update = gr.update(value=parsed_artist)
                    if parsed_title:
                        title_update = gr.update(value=parsed_title)

                # Media uploaded - enable transcribe, pulse the button, open transcription settings
                step_html = generate_step_indicator(current_step=2, completed_steps=[1], sub_steps={1: (1, 2, 2)})
                hint_html = generate_next_step_hint(has_audio=True)
                return step_html, hint_html, gr.update(interactive=True, elem_classes=["btn-ready"]), gr.update(value="*Click to analyze media*"), gr.update(open=True), artist_update, title_update

            # No media - disable transcribe
            step_html = generate_step_indicator(current_step=1, completed_steps=[], sub_steps={1: (0, 2, 1)})
            hint_html = generate_next_step_hint(has_audio=False)
            return step_html, hint_html, gr.update(interactive=False, elem_classes=[]), gr.update(value="*Upload media to enable*"), gr.update(), gr.update(), gr.update()

        audio_file.change(
            fn=on_media_upload,
            inputs=[audio_file, video_file, artist_name, song_title],
            outputs=[step_indicator, next_step_hint, transcribe_btn, transcribe_btn_hint, transcription_settings_acc, artist_name, song_title]
        )

        video_file.change(
            fn=on_media_upload,
            inputs=[audio_file, video_file, artist_name, song_title],
            outputs=[step_indicator, next_step_hint, transcribe_btn, transcribe_btn_hint, transcription_settings_acc, artist_name, song_title]
        )

        # Show and auto-check "Use Uploaded Video as Background" only when a video is uploaded
        def on_video_presence(video_path):
            if video_path:
                return gr.update(visible=True, value=True), gr.update(interactive=False)
            return gr.update(visible=False, value=False), gr.update(interactive=True)

        video_file.change(
            fn=on_video_presence,
            inputs=[video_file],
            outputs=[use_input_video, video_background]
        )

        # Show the custom background upload only when "Custom Video" is selected
        def toggle_custom_background(choice):
            return gr.update(visible=(choice == "Custom Video"))

        video_background.change(
            fn=toggle_custom_background,
            inputs=[video_background],
            outputs=[custom_background_video]
        )

        # The Background dropdown is ignored while the uploaded video is used as background
        def toggle_background_dropdown(checked):
            return gr.update(interactive=not checked)

        use_input_video.change(
            fn=toggle_background_dropdown,
            inputs=[use_input_video],
            outputs=[video_background]
        )

        # Show the Ollama model dropdown only when LLM correction is enabled
        def toggle_llm_model(checked):
            return gr.update(visible=checked)

        use_llm_correction.change(
            fn=toggle_llm_model,
            inputs=[use_llm_correction],
            outputs=[llm_corrector_model]
        )

        # Quality preset: applies model + resolution; manual overrides switch to Custom.
        # .input (user-initiated only) is used everywhere to avoid update loops.
        def apply_quality_preset(preset):
            settings = QUALITY_PRESETS.get(preset)
            if settings:
                return gr.update(value=settings["model_size"]), gr.update(value=settings["resolution"])
            return gr.update(), gr.update()

        quality_preset.input(
            fn=apply_quality_preset,
            inputs=[quality_preset],
            outputs=[model_size, resolution]
        )

        def mark_custom_preset():
            return gr.update(value="Custom")

        model_size.input(fn=mark_custom_preset, inputs=[], outputs=[quality_preset])
        resolution.input(fn=mark_custom_preset, inputs=[], outputs=[quality_preset])

        # Download from YouTube and update step
        def download_and_update_step(url, download_video):
            result = download_youtube_video(url, download_video)
            if result:
                step_html = generate_step_indicator(current_step=2, completed_steps=[1], sub_steps={1: (1, 2, 2)})
                hint_html = generate_next_step_hint(has_audio=True)
                # Video downloads go to the Video slot so they can be used as the karaoke background
                is_video = os.path.splitext(result)[1].lower() in {'.mp4', '.mkv', '.webm', '.mov'}
                audio_update = gr.update(value=None) if is_video else gr.update(value=result)
                video_update = gr.update(value=result) if is_video else gr.update(value=None)
                upload_tab = gr.update(selected=1) if is_video else gr.update(selected=0)
                return audio_update, video_update, upload_tab, step_html, hint_html, gr.update(interactive=True, elem_classes=["btn-ready"]), gr.update(value="*Click to analyze media*"), gr.update(open=True)
            else:
                step_html = generate_step_indicator(current_step=1, sub_steps={1: (0, 2, 1)})
                hint_html = generate_next_step_hint(has_audio=False)
                return gr.update(), gr.update(), gr.update(), step_html, hint_html, gr.update(interactive=False, elem_classes=[]), gr.update(value="*Upload media to enable*"), gr.update()

        download_btn.click(
            fn=download_and_update_step,
            inputs=[youtube_url, download_video_checkbox],
            outputs=[audio_file, video_file, upload_tabs, step_indicator, next_step_hint, transcribe_btn, transcribe_btn_hint, transcription_settings_acc]
        )

        # History dropdown handlers
        def update_history_dropdown():
            return gr.update(choices=get_history_choices())

        refresh_history_btn.click(
            fn=update_history_dropdown,
            inputs=[],
            outputs=[history_dropdown]
        )

        history_dropdown.change(
            fn=load_from_history,
            inputs=[history_dropdown],
            outputs=[artist_name, song_title]
        )

        # Initialize history on load
        app.load(
            fn=update_history_dropdown,
            inputs=[],
            outputs=[history_dropdown]
        )

        def process_transcription(audio, video, *args):
            # Use video if audio is not provided
            main_file = audio if audio else video
            return transcribe_audio_only(main_file, *args)

        transcribe_btn.click(
            fn=process_transcription,
            inputs=[
                audio_file, video_file, artist_name, song_title,
                use_gpu, model_size,
                use_llm_correction,
                llm_corrector_model,
                enable_beat_effects,
                use_input_video,
                lyrics_time_offset
            ],
            outputs=[error_output, preview, lyrics_display, editor_df, segments_state, beat_times_state, main_tabs, step_indicator, next_step_hint, render_btn, render_btn_hint, video_style_acc, transcribe_btn, history_dropdown]
        )

        def process_render(editor_df, audio, video, *args):
            main_file = audio if audio else video
            return render_video_from_editor(editor_df, main_file, *args)

        render_btn.click(
            fn=process_render,
            inputs=[
                editor_df, audio_file, video_file, artist_name, song_title,
                use_gpu, model_size, video_background, custom_background_video,
                font_color, resolution,
                enable_beat_effects, enable_countdown, enable_pitch_guide,
                video_effect, use_input_video, video_dimmer, vocal_volume,
                segments_state, beat_times_state
            ],
            outputs=[error_output, video_output, main_tabs, step_indicator, next_step_hint, render_btn_hint, render_btn]
        )

        gr.Markdown(
            """
            ---
            **Quick Start:** Upload audio → Transcribe → Edit lyrics & pick style → Render video. Follow the step indicator above!
            """,
            elem_id="quick-start-hint"
        )

    return app
if __name__ == "__main__":
    import uvicorn
    app = create_ui()
    try:
        uvicorn.run(
            app.app,
            host="0.0.0.0",
            port=7861,
            timeout_keep_alive=600,
            timeout_graceful_shutdown=10,
            ws_max_size=100 * 1024 * 1024
        )
    except asyncio.exceptions.CancelledError:
        logger.info("Gradio server closed gracefully")
    except Exception as e:
        logger.error(f"Gradio server error: {e}")