"""
Main entry point for the Lyrics Transcriber Gradio UI
"""

import os
import sys
import argparse

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradio_ui.app import create_ui
from gradio_ui.gpu_utils import check_cuda_availability, configure_torch_for_gpu

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Lyrics Transcriber Gradio UI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--share", action="store_true", help="Create a public link for sharing")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    args = parser.parse_args()
    
    # Check GPU availability
    has_gpu, gpu_info, vram_gb = check_cuda_availability()
    
    if has_gpu and not args.no_gpu:
        print(f"GPU detected: {gpu_info}")
        configure_torch_for_gpu(vram_gb)
    else:
        if args.no_gpu:
            print("GPU acceleration disabled by user")
        else:
            print(f"GPU not available: {gpu_info}")
    
    # Create and launch the UI
    app = create_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
