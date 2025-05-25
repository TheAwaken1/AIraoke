
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
    "Audio Particles": "audio_particles",  # NEW!
    "Black": "black"
}

VIDEO_RESOLUTIONS = {
    "360p": "360p",
    "720p": "720p",
    "1080p": "1080p",
    "4k": "4k"
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

def transcribe_lyrics(
    audio_file, artist_name, song_title,
    use_gpu, model_size, video_background,
    font_color, resolution, use_llm_correction, 
    separate_vocals, enable_beat_effects,  # No more slideshow_effect
    progress=gr.Progress()
):
    if audio_file is None:
        logger.warning("No audio file uploaded.")
        return "Please upload an audio file.", None, None, None
    is_valid, message = validate_audio_file(audio_file)
    if not is_valid:
        logger.warning(f"Invalid audio file: {message}")
        return message, None, None, None

    is_valid_content, content_message = validate_audio_content(audio_file)
    if not is_valid_content:
        logger.warning(f"Audio content validation failed: {content_message}")
        return content_message, None, None, None

    try:
        app_dir = os.path.dirname(__file__)
        output_dir = os.path.join(app_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Using output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Error creating output directory: {e}")
        return f"Error creating output directory: {e}", None, None, None
    try:
        transcriber = LyricsTranscriberWrapper(
            use_gpu=use_gpu,
            output_dir=output_dir,
            model_size=model_size,
            video_background=VIDEO_BACKGROUNDS.get(video_background, "video_1"),
            font_color=FONT_COLORS.get(font_color, "255,255,255,255"),
            resolution=VIDEO_RESOLUTIONS.get(resolution, "1080p"),
            progress_callback=progress,
            use_llm_correction=use_llm_correction,
            enable_separate_vocals=separate_vocals,
            use_beat_effects=enable_beat_effects,
        )
        if transcriber.demo_mode:
            logger.warning("Demo mode active; transcription limited.")
    except Exception as e:
        logger.error(f"Error initializing transcriber: {e}")
        return f"Error initializing transcriber: {e}", None, None, None
    def update_progress(value, description=None):
        if description:
            progress(value, desc=description)
        else:
            progress(value)
    try:
        result = transcriber.transcribe(
            audio_file,
            artist=artist_name,
            title=song_title,
            progress_callback=update_progress,
        )
    except Exception as e:
        logger.error(f"Error during transcription process: {e}", exc_info=True)
        return f"Error during transcription process: {e}", None, None, None
    if not result or not result.get("success"):
        message = result.get("message", "Unknown transcription error")
        logger.error(f"Transcription failed: {message}")
        return message, None, None, None
    lyrics_text = result.get("lyrics_text")
    video_path = result.get("video_path")
    try:
        styled_lyrics = lyrics_display_manager.format_lyrics_for_display(
            lyrics_text,
            show_upcoming=False
        )
    except Exception as e:
        logger.error(f"Error formatting lyrics display: {e}")
        styled_lyrics = "Error formatting lyrics display."
    logger.info(f"Transcription successful. Video path: {video_path}")
    return (None, lyrics_text, styled_lyrics, video_path.replace('\\', '/') if video_path else None)

def create_ui():
    has_gpu, gpu_info, vram_gb = check_cuda_availability()
    combined_css = custom_css

    with gr.Blocks(theme=theme, title="AIraoke", css=combined_css) as app:
        app.app.add_middleware(ContentLengthMiddleware)
        app.app.add_middleware(FileSizeLimitMiddleware, max_size=100 * 1024 * 1024)

        gr.Markdown(
            """
            # AIraoke
            A Gradio UI for lyrics and speech transcription with karaoke-style video playback!
            """,
            elem_id="app-title"
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["section-group", "section-upload"]):
                    gr.Markdown("### Upload Audio")
                    audio_file = gr.Audio(label="Upload Audio File", type="filepath", interactive=True)
                    with gr.Row():
                        artist_name = gr.Textbox(label="Artist Name", placeholder="Enter artist name (optional)")
                        song_title = gr.Textbox(label="Song Title", placeholder="Enter song title (optional)")

                with gr.Accordion("Advanced Settings", open=False, elem_classes=["section-accordion", "section-settings"]):
                    with gr.Row():
                        use_gpu = gr.Checkbox(
                            label=f"Use GPU Acceleration {' (Available)' if has_gpu else ' (Not Available)'}",
                            value=has_gpu,
                            interactive=has_gpu
                        )
                        model_size = gr.Dropdown(
                            label="Whisper Model",
                            choices=["turbo", "large-v3"],
                            value="large-v3"
                        )
                    with gr.Row():
                        video_background = gr.Dropdown(
                            label="Video Background",
                            choices=list(VIDEO_BACKGROUNDS.keys()),
                            value="Video 1"
                        )
                    with gr.Row():
                        font_color = gr.Dropdown(
                            label="Font Color",
                            choices=list(FONT_COLORS.keys()),
                            value="Fire Red"
                        )
                        resolution = gr.Dropdown(
                            label="Video Resolution",
                            choices=list(VIDEO_RESOLUTIONS.keys()),
                            value="1080p"
                        )
                    with gr.Row():
                        use_llm_correction = gr.Checkbox(
                            label="Use LLM for Lyrics Correction",
                            value=False
                        )
                        separate_vocals = gr.Checkbox(
                            label="Separate Vocals for Karaoke",
                            value=False
                        )
                    with gr.Row():
                        enable_beat_effects = gr.Checkbox(
                            label="Enable Beat Effects",
                            value=True,  # Default to True for particles
                            info="Add beat-synchronized effects (explosions for particles)"
                        )
                    if not has_gpu:
                        gr.Markdown(f"*GPU Status: {gpu_info}*")

                transcribe_btn = gr.Button("Transcribe Lyrics", variant="primary")

            with gr.Column(scale=2, elem_classes=["section-outputs"]):
                with gr.Tabs():
                    with gr.TabItem("Preview"):
                        preview = gr.Textbox(
                            label="Lyrics Preview",
                            placeholder="Transcribed lyrics will appear here...",
                            lines=20,
                            interactive=False
                        )
                    with gr.TabItem("Lyrics Display"):
                        lyrics_display = gr.HTML("Synchronized lyrics will appear here after transcription.")
                    with gr.TabItem("Video Playback"):
                        video_output = gr.Video(label="Lyrics Video", interactive=False)

                error_output = gr.Textbox(label="Status", visible=False, interactive=False)

        transcribe_btn.click(
            fn=transcribe_lyrics,
            inputs=[
                audio_file, artist_name, song_title,
                use_gpu, model_size, video_background,
                font_color, resolution, use_llm_correction, 
                separate_vocals, enable_beat_effects  # No more slideshow_effect
            ],
            outputs=[error_output, preview, lyrics_display, video_output]
        )

        gr.Markdown(
            """
            ### How to Use
            1. Upload an audio file (MP3, WAV, etc.).
            2. Optionally enter Artist Name and Song Title for output naming.
            3. Adjust advanced settings:
                - **Use GPU Acceleration:** Faster processing if GPU is available.
                - **Whisper Model:** Turbo (fast) or large-v3 (accurate, needs ~10GB memory).
                - **Video Background:** Choose a background style for the karaoke video (add your own videos to app/gradio_ui/backgrounds, make sure to use the exact mp4 name).
                - **Font Color:** Choose the color of the lyrics text.
                - **Video Resolution:** Choose the resolution of the output video (360p, 720p, 1080p, 4k).
                - **Use LLM for Lyrics Correction:** Enable LLM to correct lyrics (adds ~1-2s) (lyrics correction is not perfect, it misses entire sections of lyrics with some songs)
                - **Separate Vocals for Karaoke:** Remove vocals for a karaoke-style video.
                - **Enable Beat Effects:** Works with Audio Partiles video Background (marked as default does not affect video).
            4. Click "Transcribe Lyrics" to process.
            5. View results in tabs; video and files will be saved to the output directory.
            Note: Uses Whisper for transcription, with optional LLM correction and Demucs for vocal separation.
            ---
            Gradio UI Made by TheAwakenOne
            """
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