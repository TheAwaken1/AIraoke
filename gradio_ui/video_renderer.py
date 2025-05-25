import subprocess
import os
import logging
import shlex
import stat
import json
import tempfile
from typing import List, Optional
try:
    import librosa
except ImportError:
    librosa = None

logger = logging.getLogger(__name__)

def get_ffmpeg_path() -> str:
    binary_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    app_base = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(app_base, "..", "bin", binary_name),
        os.path.join(app_base, "bin", binary_name),
        os.path.join(app_base, "ffmpeg", binary_name),
        os.path.join(app_base, binary_name),
    ]
    for ffmpeg_path in possible_paths:
        if os.path.exists(ffmpeg_path):
            try:
                subprocess.run([ffmpeg_path, "-version"], capture_output=True, check=True)
                return ffmpeg_path
            except:
                continue
    return binary_name

def get_ffprobe_path() -> str:
    binary_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    app_base = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(app_base, "..", "bin", binary_name),
        os.path.join(app_base, "bin", binary_name),
        os.path.join(app_base, "ffmpeg", binary_name),
        os.path.join(app_base, binary_name),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                subprocess.run([path, "-version"], capture_output=True, check=True)
                return path
            except:
                continue
    return binary_name

def get_audio_duration(filepath: str) -> Optional[float]:
    ffprobe_path = get_ffprobe_path()
    if not ffprobe_path:
        logger.error("No valid ffprobe path found.")
        return None

    cmd = [
        ffprobe_path, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        filepath
    ]

    try:
        result = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        logger.debug(f"Audio duration for {filepath}: {duration} seconds")
        return duration
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe error getting duration: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting audio duration: {e}")
        return None

def detect_beats(audio_filepath: str) -> List[float]:
    if librosa is None:
        logger.error("librosa not available, cannot detect beats. Returning empty beat list.")
        return []
    
    try:
        y, sr = librosa.load(audio_filepath, sr=None)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        logger.info(f"Detected {len(beat_times)} beats with estimated tempo {tempo:.2f} BPM")
        return beat_times.tolist()
    except Exception as e:
        logger.error(f"Error detecting beats: {e}")
        return []

def check_nvenc_support():
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        logger.error("No valid FFmpeg path found.")
        return False
    
    logger.debug(f"Checking ffmpeg path: {ffmpeg_path}")
    if os.path.exists(ffmpeg_path):
        logger.debug(f"ffmpeg found at {ffmpeg_path}")
        try:
            file_stats = os.stat(ffmpeg_path)
            logger.debug(f"ffmpeg permissions: {oct(file_stats.st_mode & 0o777)}")
            if not (file_stats.st_mode & stat.S_IXUSR):
                logger.warning(f"ffmpeg at {ffmpeg_path} is not executable")
        except Exception as e:
            logger.warning(f"Could not check ffmpeg file stats: {e}")
    else:
        logger.warning(f"ffmpeg not found at {ffmpeg_path}")

    cmd = [ffmpeg_path, "-encoders"]
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return "h264_nvenc" in result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking NVENC support: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"ffmpeg not found at {ffmpeg_path}. Ensure FFmpeg is installed in app folder or system PATH.")
        return False

def build_beat_brightness_filter(
        beat_times: List[float],
        duration: float = 0.30,
        base: float = -0.15,
        pulse: float = 0.45
) -> str:
    """
    Return an ffmpeg 'eq' filter that adds a brightness pulse on each beat.

    brightness(t) = base + pulse * [ 1 while t falls in ANY beat window ]
    """
    max_beats = 50                          # keep the filter string reasonable
    windows = [
        f"between(t\\,{t:.3f}\\,{t+duration:.3f})"
        for t in beat_times[:max_beats]
    ]
    # OR all beat windows by summing them, then test gt(sum,0)
    sum_windows = "+".join(windows)         # 0 ➜ dark, ≥1 ➜ pulse
    return (
        "eq=brightness='({base})+({pulse})*gt(({sum_windows})\\,0)'"
        .format(base=base, pulse=pulse, sum_windows=sum_windows)
    )

def render_video_with_background(audio_filepath, ass_filepath, output_filepath, 
                                background_image=None, resolution="1080p", 
                                beat_times=None):
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        logger.error("No valid FFmpeg path found.")
        return None

    # Check NVENC support
    use_nvenc = check_nvenc_support()
    logger.info(f"NVENC support: {'Available' if use_nvenc else 'Not available'}")

    logger.debug(f"Checking ffmpeg path: {ffmpeg_path}")
    if os.path.exists(ffmpeg_path):
        logger.debug(f"ffmpeg found at {ffmpeg_path}")
        try:
            file_stats = os.stat(ffmpeg_path)
            logger.debug(f"ffmpeg permissions: {oct(file_stats.st_mode & 0o777)}")
            if not (file_stats.st_mode & stat.S_IXUSR):
                logger.warning(f"ffmpeg at {ffmpeg_path} is not executable")
        except Exception as e:
            logger.warning(f"Could not check ffmpeg file stats: {e}")
    else:
        logger.warning(f"ffmpeg not found at {ffmpeg_path}")

    safe_audio_filepath = audio_filepath.replace('\\', '/')
    safe_ass_filepath = ass_filepath.replace('\\', '/').replace(':', '\\:')
    safe_output_filepath = output_filepath.replace('\\', '/')

    resolution_map = {
        "360p": (640, 360),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "4k": (3840, 2160)
    }
    if resolution not in resolution_map:
        logger.warning(f"Invalid resolution '{resolution}'. Defaulting to 1080p.")
        resolution = "1080p"
    width, height = resolution_map[resolution]

    filter_complex = []
    background_path = None
    use_background = False

    audio_duration = get_audio_duration(audio_filepath)
    logger.info(f"Audio duration: {audio_duration}s")
    if audio_duration is None:
        logger.error("Cannot render – audio duration unknown.")
        return None

    if beat_times is None:
        beat_times = detect_beats(audio_filepath)
    if not beat_times:
        logger.warning("No valid beat timestamps provided or detected. Beat effects will be disabled.")
        beat_times = []

    # Handle Audio Particles background
    if background_image == "audio_particles":
        # Generate particle visualization
        particle_video_path = os.path.join(
            os.path.dirname(output_filepath), 
            "particle_visualization.mp4"
        )
        
        logger.info("Creating audio particle visualization...")
        
        try:
            # Import the particle visualizer
            from gradio_ui.audio_particles import AudioParticleVisualizer
            
            # Create visualizer with beat support
            visualizer = AudioParticleVisualizer(
                audio_filepath, 
                resolution=resolution,
                use_beats=bool(beat_times)
            )
            
            # If beat_times were passed, use them
            if beat_times:
                import numpy as np
                visualizer.beat_times = np.array(beat_times)
                visualizer.beats = librosa.time_to_frames(beat_times, sr=visualizer.sr)
            
            # Generate the particle video
            background_path = visualizer.generate_video(particle_video_path)
            
            if background_path and os.path.exists(background_path):
                use_background = True
                logger.info(f"Using particle visualization: {background_path}")
            else:
                logger.error("Particle visualization failed. Falling back to black background.")
                background_path = None
                use_background = False
                
        except Exception as e:
            logger.error(f"Error creating particle visualization: {e}")
            background_path = None
            use_background = False
            
    elif background_image and os.path.exists(background_image):
        background_path = background_image
        use_background = True
        logger.info(f"Using background video: {background_path}")

    # Build filter complex
    filter_complex = []
    if use_background:
        filter_complex.append(
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1,fps=30[bg]"
        )
    else:
        filter_complex.append(
            f"color=c=black:s={width}x{height}:d={audio_duration}:r=30[bg]"
        )

    # Apply beat effects only for regular video backgrounds (not particles)
    if beat_times and background_image != "audio_particles":
        brightness_filter = build_beat_brightness_filter(beat_times, duration=0.30)
        filter_complex.append(f"[bg]{brightness_filter}[bg_beat]")
        bg_src = "[bg_beat]"
    else:
        bg_src = "[bg]"

    safe_ass = ass_filepath.replace('\\', '/').replace(':', '\\:')
    filter_complex.append(f"{bg_src}subtitles='{safe_ass}'[out_v]")

    filter_complex_str = ";".join(filter_complex)

    # Build FFmpeg command
    cmd = [
        ffmpeg_path, "-y",
    ]
    if use_background:
        cmd.extend(["-stream_loop", "-1", "-i", background_path.replace('\\', '/')])
        cmd.extend(["-i", safe_audio_filepath])
        input_count = 2
    else:
        cmd.extend(["-i", safe_audio_filepath])
        input_count = 1
    
    cmd.extend([
        "-filter_complex", filter_complex_str,
        "-map", "[out_v]",
        "-map", f"{input_count-1}:a",
    ])
    
    if use_background:
        cmd.extend(["-map", "-0:a"])
    
    cmd.extend([
        "-c:v", "h264_nvenc" if use_nvenc else "libx264",
        "-preset", "p4" if use_nvenc else "medium",
    ])
    
    if use_nvenc:
        cmd.extend(["-rc", "vbr", "-b:v", "8M"])
    else:
        cmd.extend(["-crf", "23"])
    
    cmd.extend([
        "-c:a", "copy",
        "-shortest",
        safe_output_filepath
    ])

    logger.debug(f"Running FFmpeg command: {' '.join(shlex.quote(str(c)) for c in cmd)}")

    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', timeout=1800)
        logger.info(f"Successfully rendered video: {safe_output_filepath}")
        logger.debug(f"FFmpeg stdout: {result.stdout}")
        logger.info(f"FFmpeg stderr: {result.stderr}")
        return output_filepath
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error during rendering: {shlex.join(e.cmd)}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Stderr: {e.stderr}")
        logger.error(f"Stdout: {e.stdout}")
        return None
    except subprocess.TimeoutExpired as e:
        logger.error(f"FFmpeg command timed out after {e.timeout} seconds.")
        logger.error(f"Command: {shlex.join(e.cmd)}")
        logger.error(f"Stderr output before timeout: {e.stderr}")
        logger.error(f"Stdout output before timeout: {e.stdout}")
        return None
    except FileNotFoundError:
        logger.error(f"ffmpeg not found at {ffmpeg_path}. Ensure FFmpeg is installed in app folder or system PATH.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during FFmpeg execution: {e}")
        logger.error(f"Command attempted: {' '.join(shlex.quote(str(c)) for c in cmd)}")
        return None