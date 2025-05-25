"""
Utility functions for the Lyrics Transcriber Gradio UI
"""

import os
import tempfile
import shutil
from pathlib import Path
import time

def create_output_directory():
    """Create output directory for transcribed files if it doesn't exist"""
    output_dir = Path.home() / "lyrics_transcriber_output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def validate_audio_file(file_path):
    """
    Validate that the uploaded file is a supported audio format
    Returns: (is_valid, message)
    """
    if file_path is None:
        return False, "No file uploaded"
    
    supported_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext not in supported_extensions:
        return False, f"Unsupported file format. Please upload one of: {', '.join(supported_extensions)}"
    
    # Check if file exists and is readable
    try:
        with open(file_path, 'rb') as f:
            # Just read a small chunk to verify file is accessible
            f.read(1024)
        return True, "File is valid"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def format_time(seconds):
    """Format seconds as MM:SS.ms"""
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"

def create_temp_copy(file_path):
    """Create a temporary copy of the file to work with"""
    temp_dir = tempfile.gettempdir()
    file_name = os.path.basename(file_path)
    temp_path = os.path.join(temp_dir, f"lyrics_transcriber_{file_name}")
    
    shutil.copy2(file_path, temp_path)
    return temp_path

def simulate_countdown(seconds, progress_callback=None):
    """
    Simulate a countdown timer
    This is a placeholder that will be replaced with actual implementation
    """
    if progress_callback:
        for i in range(seconds, 0, -1):
            progress_callback(i)
            time.sleep(1)
    else:
        time.sleep(seconds)

def format_lyrics_for_display(lyrics_data, background_style=None, show_upcoming=True):
    """
    Format lyrics data for display in the UI
    This is a placeholder that will be replaced with actual implementation
    """
    # Placeholder implementation
    if not lyrics_data:
        return "No lyrics data available"
    
    if isinstance(lyrics_data, str):
        return lyrics_data
    
    # In the real implementation, this would format the lyrics with proper timing
    # and apply styling based on the background_style parameter
    formatted_html = f"<div class='lyrics-container'>{lyrics_data}</div>"
    
    return formatted_html
