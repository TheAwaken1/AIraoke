"""
Enhanced features implementation for the Lyrics Transcriber Gradio UI
"""

import os
import time
import threading
import logging
from pathlib import Path
import base64
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LyricsDisplayManager:
    """Manager for enhanced lyrics display features"""
    
    def __init__(self):
        """Initialize the lyrics display manager"""
        self.background_layouts = {
            "gradient_blue": self._get_gradient_blue_style(),
            "gradient_purple": self._get_gradient_purple_style(),
            "starry_night": self._get_starry_night_style(),
            "minimal_dark": self._get_minimal_dark_style(),
            "minimal_light": self._get_minimal_light_style(),
            "karaoke_stage": self._get_karaoke_stage_style(),
            "music_notes": self._get_music_notes_style(),
        }
    
    # --- Make sure the format_lyrics_for_display method is also in this file ---
    # (It should be there already based on previous context)
    def format_lyrics_for_display(self, lyrics_text, background_style=None, show_upcoming=True):
        """
        Format lyrics text for display with enhanced styling

        Args:
            lyrics_text: Raw lyrics text with timestamps
            background_style: Background style name
            show_upcoming: Whether to show upcoming lyrics (will be False from app.py)

        Returns:
            HTML-formatted lyrics for display
        """
        if not lyrics_text:
            return "<div class='lyrics-display'>No lyrics available</div>"

        segments = self._parse_lyrics_segments(lyrics_text) # Assuming _parse_lyrics_segments exists

        html_content = ""
        for i, segment in enumerate(segments):
            # Simplified: just output the text line by line
            # The dynamic highlighting requires JavaScript not implemented here
            html_content += f"<div>{segment['text']}</div>\n"
            # --- REMOVED distinction for current/upcoming ---
            # if i == 0:
            #     html_content += f"<div class='current-lyric'>{segment['text']}</div>\n"
            # elif show_upcoming: # This will be False
            #     html_content += f"<div class='upcoming-lyric'>{segment['text']}</div>\n"
            # else:
            #     html_content += f"<div>{segment['text']}</div>\n"
            # --- END REMOVAL ---

        styled_html = self.apply_background_style(html_content, background_style) # Assuming apply_background_style exists

        return styled_html
    
    def apply_background_style(self, lyrics_html, background_style):
        """Apply background style to lyrics HTML"""
        if background_style is None or background_style == "None" or background_style not in self.background_layouts:
            return f"<div class='lyrics-display'>{lyrics_html}</div>"
        
        style = self.background_layouts.get(background_style, "")
        
        return f"""
        <div class='lyrics-display bg-{background_style}' style='{style}'>
            {lyrics_html}
        </div>
        """
    
    def create_countdown_html(self, seconds):
        """Create HTML for countdown timer"""
        return f"""
        <div class='countdown-container'>
            <div class='countdown'>{seconds}</div>
            <div class='countdown-text'>Starting in...</div>
        </div>
        """
    
    def highlight_current_lyric(self, lyrics_html, current_index, show_upcoming=True):
        """Highlight the current lyric and optionally style upcoming lyrics"""
        # This would be implemented with JavaScript in a real-time application
        # For now, we'll just return the lyrics HTML as is
        return lyrics_html
    
    def _parse_lyrics_segments(self, lyrics_text):
        """Parse lyrics text with timestamps into segments"""
        segments = []
        
        # Regular expression to match timestamp format [MM:SS.ms - MM:SS.ms]
        pattern = r'\[(\d+:\d+\.\d+) - (\d+:\d+\.\d+)\] (.*)'
        
        for line in lyrics_text.strip().split('\n'):
            match = re.match(pattern, line)
            if match:
                start_time, end_time, text = match.groups()
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text
                })
            elif line.strip():  # If there's text but no timestamp
                segments.append({
                    'start_time': '00:00.00',
                    'end_time': '00:00.00',
                    'text': line.strip()
                })
        
        return segments
    
    # --- Add text-shadow to these styles for better contrast ---

    def _get_gradient_blue_style(self):
        """Get CSS style for gradient blue background"""
        # Added text-shadow
        return "background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%); color: white; padding: 20px; border-radius: 10px; text-shadow: 1px 1px 2px black;"

    def _get_gradient_purple_style(self):
        """Get CSS style for gradient purple background"""
        # Added text-shadow
        return "background: linear-gradient(135deg, #8b5cf6 0%, #4c1d95 100%); color: white; padding: 20px; border-radius: 10px; text-shadow: 1px 1px 2px black;"

    def _get_starry_night_style(self):
        """Get CSS style for starry night background"""
        # Added text-shadow
        return "background-color: #0f172a; background-image: radial-gradient(white, rgba(255, 255, 255, 0.2) 2px, transparent 2px); background-size: 50px 50px; color: white; padding: 20px; border-radius: 10px; text-shadow: 1px 1px 2px black;"

    def _get_minimal_dark_style(self):
        """Get CSS style for minimal dark background"""
        # White text on dark - usually okay, but shadow doesn't hurt
        # Added text-shadow
        return "background-color: #1e293b; color: #f8fafc; padding: 20px; border-radius: 10px; text-shadow: 1px 1px 2px black;"

    def _get_minimal_light_style(self):
        """Get CSS style for minimal light background"""
        # Dark text on light - should be fine, no shadow needed
        return "background-color: #f8fafc; color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0;"

    def _get_karaoke_stage_style(self):
        """Get CSS style for karaoke stage background"""
        # Already had text-shadow, maybe make it stronger? Adjusted alpha
        return "background: linear-gradient(to bottom, #000000 0%, #434343 100%); color: #f8fafc; text-shadow: 0 0 5px rgba(0, 0, 0, 0.8); padding: 20px; border-radius: 10px;"

    def _get_music_notes_style(self):
        """Get CSS style for music notes background"""
        # Dark text on light - should be fine, no shadow needed
        music_note_svg = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9InJnYmEoMCwwLDAsMC4xKSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik05IDE4VjVsMTItMnYxMyI+PC9wYXRoPjxjaXJjbGUgY3g9IjYiIGN5PSIxOCIgcj0iMyI+PC9jaXJjbGU+PGNpcmNsZSBjeD0iMTgiIGN5PSIxNiIgcj0iMyI+PC9jaXJjbGU+PC9zdmc+"
        return f"background-color: #f8fafc; background-image: url('{music_note_svg}'); background-size: 100px 100px; color: #1e293b; padding: 20px; border-radius: 10px;"

class CountdownManager:
    """Manager for countdown timer functionality"""
    
    def __init__(self):
        """Initialize the countdown manager"""
        self.countdown_active = False
        self.countdown_thread = None
    
    def start_countdown(self, seconds, callback=None):
        """
        Start a countdown timer
        
        Args:
            seconds: Number of seconds to count down
            callback: Function to call with each countdown update
            
        Returns:
            True if countdown started, False otherwise
        """
        if self.countdown_active:
            return False
        
        self.countdown_active = True
        self.countdown_thread = threading.Thread(
            target=self._countdown_worker,
            args=(seconds, callback),
            daemon=True
        )
        self.countdown_thread.start()
        return True
    
    def stop_countdown(self):
        """Stop the countdown timer"""
        self.countdown_active = False
        if self.countdown_thread and self.countdown_thread.is_alive():
            self.countdown_thread.join(timeout=1.0)
    
    def _countdown_worker(self, seconds, callback):
        """Worker function for countdown thread"""
        try:
            for i in range(seconds, 0, -1):
                if not self.countdown_active:
                    break
                
                if callback:
                    callback(i)
                
                time.sleep(1)
            
            # Final callback with 0
            if self.countdown_active and callback:
                callback(0)
        
        except Exception as e:
            logger.error(f"Error in countdown worker: {e}")
        finally:
            self.countdown_active = False

class UpcomingLyricsPreview:
    """Manager for upcoming lyrics preview functionality"""
    
    def __init__(self, preview_lines=3):
        """
        Initialize the upcoming lyrics preview manager
        
        Args:
            preview_lines: Number of upcoming lines to preview
        """
        self.preview_lines = preview_lines
    
    def format_with_preview(self, lyrics_text, current_line=0):
        """
        Format lyrics with preview of upcoming lines
        
        Args:
            lyrics_text: Raw lyrics text
            current_line: Current line index
            
        Returns:
            Formatted lyrics with preview
        """
        if not lyrics_text:
            return "No lyrics available"
        
        lines = lyrics_text.strip().split('\n')
        if current_line >= len(lines):
            current_line = len(lines) - 1
        
        # Format current line
        formatted = f"<div class='current-lyric'>{lines[current_line]}</div>\n"
        
        # Add preview of upcoming lines
        for i in range(1, self.preview_lines + 1):
            if current_line + i < len(lines):
                formatted += f"<div class='upcoming-lyric'>{lines[current_line + i]}</div>\n"
        
        return formatted
