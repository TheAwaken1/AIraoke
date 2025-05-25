"""
Styles and CSS for the Lyrics Transcriber Gradio UI
"""

def get_css():
    """Return custom CSS for the Gradio UI"""
    return """
    /* Main container styling */
    .gradio-container {
        max-width: 1200px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    /* Header styling */
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
        color: white !important;
        text-align: center !important;
        font-weight: 700 !important;
    }
    
    /* Button styling */
    button.primary {
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        transition: all 0.2s ease !important;
    }
    
    button.primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Group and accordion styling */
    .gradio-group, .gradio-accordion {
        border-radius: 0.75rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    }
    
    /* Tab styling */
    .gradio-tabitem {
        padding: 1rem !important;
    }
    
    /* Lyrics display styling */
    .lyrics-display {
        font-family: 'Arial', sans-serif !important;
        line-height: 1.6 !important;
        padding: 1.5rem !important;
        border-radius: 0.5rem !important;
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        max-height: 500px !important;
        overflow-y: auto !important;
    }
    
    /* Background layouts styling */
    .bg-gradient-blue {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%) !important;
        color: white !important;
    }
    
    .bg-gradient-purple {
        background: linear-gradient(135deg, #8b5cf6 0%, #4c1d95 100%) !important;
        color: white !important;
    }
    
    /* Highlighted lyrics styling */
    .current-lyric {
        font-weight: bold !important;
        color: #4338ca !important;
        font-size: 1.2em !important;
        text-shadow: 0 0 5px rgba(67, 56, 202, 0.3) !important;
    }
    
    .upcoming-lyric {
        color: #6b7280 !important;
        font-style: italic !important;
    }
    
    /* Countdown styling */
    .countdown {
        font-size: 3rem !important;
        font-weight: bold !important;
        color: #4338ca !important;
        text-align: center !important;
        margin: 2rem 0 !important;
        text-shadow: 0 0 10px rgba(67, 56, 202, 0.5) !important;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .gradio-container {
            padding: 0.5rem !important;
        }
        
        h1 {
            font-size: 1.8rem !important;
        }
        
        button.primary {
            font-size: 1rem !important;
            padding: 0.5rem 1rem !important;
        }
    }
    """

def apply_background_style(lyrics_html, background_style):
    """Apply background style to lyrics HTML"""
    if background_style == "None" or background_style is None:
        return lyrics_html
    
    bg_class = ""
    if background_style == "gradient_blue":
        bg_class = "bg-gradient-blue"
    elif background_style == "gradient_purple":
        bg_class = "bg-gradient-purple"
    elif background_style == "starry_night":
        bg_class = "bg-starry-night"
    elif background_style == "minimal_dark":
        bg_class = "bg-minimal-dark"
    elif background_style == "minimal_light":
        bg_class = "bg-minimal-light"
    elif background_style == "karaoke_stage":
        bg_class = "bg-karaoke-stage"
    elif background_style == "music_notes":
        bg_class = "bg-music-notes"
    
    return f'<div class="lyrics-display {bg_class}">{lyrics_html}</div>'

def highlight_current_lyric(lyrics_html, current_index, show_upcoming=True):
    """Highlight the current lyric and optionally style upcoming lyrics"""
    # This is a placeholder function that will be implemented in step 006
    return lyrics_html

def create_countdown_html(seconds):
    """Create HTML for countdown timer"""
    return f'<div class="countdown">{seconds}</div>'
