"""
Initialize the Gradio UI package
"""

from gradio_ui.app import create_ui
from gradio_ui.transcriber import LyricsTranscriberWrapper
from gradio_ui.gpu_utils import check_cuda_availability, configure_torch_for_gpu
