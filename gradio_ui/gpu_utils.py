"""
GPU detection and acceleration support for the Lyrics Transcriber
"""

import os
import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_cuda_availability():
    """
    Check if CUDA is available and return GPU information
    Returns: (is_available, info_message, vram_gb)
    """
    try:
        # Try to import torch to check CUDA availability
        import torch
        
        if torch.cuda.is_available():
            # Get device information
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            
            # Try to get VRAM information
            try:
                # This is a rough estimate and may not be accurate for all GPUs
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                vram_gb = round(vram_bytes / (1024**3), 2)  # Convert to GB
                
                # Check if VRAM meets minimum requirements
                if vram_gb >= 8:
                    vram_status = "sufficient"
                    if vram_gb >= 12:
                        vram_status = "optimal"
                else:
                    vram_status = "insufficient"
                
                info = f"Found {device_count} GPU(s). Using: {device_name} with {vram_gb} GB VRAM ({vram_status})"
                return True, info, vram_gb
            except Exception as e:
                logger.warning(f"Could not determine VRAM size: {e}")
                info = f"Found {device_count} GPU(s). Using: {device_name} (VRAM size unknown)"
                return True, info, None
        else:
            return False, "CUDA is not available. Using CPU mode.", 0
            
    except ImportError:
        logger.warning("PyTorch is not installed. Cannot check CUDA availability.")
        return False, "PyTorch is not installed. Using CPU mode.", 0
    except Exception as e:
        logger.error(f"Error checking CUDA availability: {e}")
        return False, f"Error checking GPU: {str(e)}", 0

def get_optimal_model_size(vram_gb):
    """
    Determine the optimal Whisper model size based on available VRAM
    """
    if vram_gb is None:
        # If VRAM size is unknown, default to medium
        return "medium"
    
    # Model size recommendations based on VRAM
    # Optimized specifically for 8GB and 12GB VRAM as requested
    if vram_gb >= 16:
        return "large"
    elif vram_gb >= 12:
        # Optimized for 12GB VRAM
        return "medium.en" if is_english_audio() else "medium"
    elif vram_gb >= 8:
        # Optimized for 8GB VRAM
        return "small.en" if is_english_audio() else "small"
    elif vram_gb >= 4:
        return "base.en" if is_english_audio() else "base"
    else:
        return "tiny.en" if is_english_audio() else "tiny"

def is_english_audio():
    """
    Placeholder function to determine if audio is in English
    In a real implementation, this would analyze the audio or use user input
    """
    # Default to False to be safe (use multilingual models)
    return False

def configure_torch_for_gpu(vram_gb):
    """
    Configure PyTorch settings for optimal GPU usage based on available VRAM
    Specifically optimized for 8GB and 12GB VRAM configurations, with new tier for 24GB+
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.info("CUDA not available, skipping GPU configuration")
            return False
        
        # Set PyTorch to use the first CUDA device
        torch.cuda.set_device(0)
        
        # Configure memory usage based on available VRAM
        if vram_gb is not None:
            # Optimize settings based on VRAM size
            if vram_gb >= 24:
                # For 24GB+ VRAM GPUs
                reserved_memory = 1.0  # GB
                batch_size = 32
                logger.info(f"Configuring for 24GB+ VRAM GPU with batch size {batch_size}")
            elif vram_gb >= 12:
                # For 12-24GB VRAM GPUs
                reserved_memory = 1.0  # GB
                batch_size = 16
                logger.info(f"Configuring for 12-24GB VRAM GPU with batch size {batch_size}")
            elif vram_gb >= 8:
                # For 8-12GB VRAM GPUs
                reserved_memory = 1.5  # GB
                batch_size = 8
                logger.info(f"Configuring for 8-12GB VRAM GPU with batch size {batch_size}")
            else:
                # For smaller VRAM GPUs
                reserved_memory = 0.5  # GB
                batch_size = 4
                logger.info(f"Configuring for smaller VRAM GPU with batch size {batch_size}")
            
            usable_memory = max(1.0, vram_gb - reserved_memory)
            
            # Set maximum memory usage (this is approximate)
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                memory_fraction = usable_memory / vram_gb
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                logger.info(f"Set PyTorch memory fraction to {memory_fraction:.2f}")
            
            # Store batch size in a global variable for the transcriber to use
            global OPTIMAL_BATCH_SIZE
            OPTIMAL_BATCH_SIZE = batch_size
            
        # Enable TF32 precision on Ampere or newer GPUs for better performance
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 precision for compatible GPUs")
        
        # Enable mixed precision for better performance
        if vram_gb >= 8:
            # Mixed precision works well with 8GB+ VRAM
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN benchmark mode for better performance")
        
        return True
    
    except Exception as e:
        logger.error(f"Error configuring GPU settings: {e}")
        return False

# Global variable to store optimal batch size
OPTIMAL_BATCH_SIZE = 8  # Default value

def install_pytorch_with_cuda():
    """
    Install PyTorch with CUDA support
    This function would be used in a production environment to ensure PyTorch is installed
    """
    try:
        # Check if CUDA is installed on the system
        nvidia_smi_output = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if nvidia_smi_output.returncode != 0:
            logger.warning("NVIDIA driver not found. Installing PyTorch CPU version.")
            cuda_version = None
        else:
            # Parse CUDA version from nvidia-smi output
            import re
            match = re.search(r"CUDA Version: (\d+\.\d+)", nvidia_smi_output.stdout)
            if match:
                cuda_version = match.group(1)
                logger.info(f"Detected CUDA version: {cuda_version}")
            else:
                logger.warning("Could not determine CUDA version. Installing PyTorch CPU version.")
                cuda_version = None
        
        # Install appropriate PyTorch version
        if cuda_version:
            major_version = int(cuda_version.split('.')[0])
            if major_version >= 11:
                # For CUDA 11.x or newer
                cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"]
            elif major_version == 10:
                # For CUDA 10.x
                cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu102"]
            else:
                # Fallback to CPU version for unsupported CUDA versions
                cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
        else:
            # CPU version
            cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
        
        logger.info(f"Installing PyTorch with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("PyTorch installation successful")
            return True, "PyTorch installation successful"
        else:
            logger.error(f"PyTorch installation failed: {result.stderr}")
            return False, f"PyTorch installation failed: {result.stderr}"
    
    except Exception as e:
        logger.error(f"Error installing PyTorch: {e}")
        return False, f"Error installing PyTorch: {str(e)}"