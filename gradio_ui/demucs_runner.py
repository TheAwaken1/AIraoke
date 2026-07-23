"""Entry point for Demucs vocal separation with a torchaudio save fallback.

torchaudio 2.9+ delegates torchaudio.save() to the optional torchcodec
package. When torchcodec is not installed, Demucs finishes separating but
crashes while writing the stems. To avoid depending on torchcodec (which
also requires FFmpeg shared libraries), stems are written through soundfile
whenever torchcodec is unavailable.
"""
import sys


def ensure_torchaudio_save_backend():
    """Replace torchaudio.save with a soundfile writer if torchcodec is missing."""
    try:
        from torchcodec.encoders import AudioEncoder  # noqa: F401
        return  # native torchaudio.save works
    except Exception:
        pass

    import torchaudio

    def save_with_soundfile(uri, src, sample_rate, **kwargs):
        import soundfile as sf
        # Demucs passes (channels, time) tensors; soundfile expects (time, channels)
        sf.write(str(uri), src.detach().cpu().numpy().T, int(sample_rate), subtype="PCM_16")

    torchaudio.save = save_with_soundfile


if __name__ == "__main__":
    ensure_torchaudio_save_backend()
    from demucs.separate import main
    sys.argv = ["demucs"] + sys.argv[1:]
    main()
