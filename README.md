# 🎤 AIraoke

<p align="center">
  <img src="icon.png" alt="AIraoke Logo" width="50%"/>
</p>

<a href="https://buymeacoffee.com/cmpcreativn" target="_blank">
  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="217" height="60">
</a>

**AIraoke** is a fun, experimental app that brings lyric transcriptions to life as karaoke-style MP4 videos. Built on Python-Lyric-Transcriber’s transcription logic, it adds a custom Gradio UI to visualize lyrics with beat-synced effects. Running locally on PC, Linux, or Mac without APIs, it uses Whisper for transcription, an LLM for optional lyric tweaks, and Demucs for vocal separation. Outputs may not be perfect, but it’s a creative playground for karaoke lovers, creators, and audio enthusiasts!

---

## 🚀 Features

- 🪜 **Guided 4-Step Workflow** – Upload → Transcribe → Edit → Render, with a step indicator, contextual hints, and settings that open automatically as you progress
- 🎧 **Audio Transcription** using OpenAI Whisper (Turbo / Large-v3)
- ⚡ **Quality Presets** – Fast / Balanced / Best one-click presets (switches to Custom when you fine-tune the model or resolution yourself)
- 📺 **YouTube Downloads** – Paste a link to grab audio, or the full MP4 to use as your karaoke background
- 🎬 **Backgrounds Your Way** – Bundled videos, your own uploaded/custom video (loops automatically), Audio Particles, or plain black
- ✨ **Audio Particles Visualizer** – Neon bloom, a rotating spectrum ring driven by the music, beat shockwaves, and a bass-pulsing background
- 🔊 **Vocal Volume Slider** – 100% keeps original vocals, lower it for quieter vocals, 0% for a pure instrumental (Demucs separation runs automatically)
- ✏️ **Lyrics Editor** – Fix words and timing in a table before rendering
- 🧠 **LLM Correction** (optional) – Use a local Ollama model to clean up lyrics
- 🕺 **Beat Effects, Countdown & Video Effects** – Beat-synced brightness, a 3-2-1-GO! countdown, plus Black & White / Sepia / Vignette / Blur / Invert and a background dimmer
- 🕘 **Song History** – Recent songs are saved automatically (artist/title auto-filled from the filename) and reloadable from a dropdown
- 🎥 **Karaoke Video Output** – Watch or download from the Video Playback tab (360p to 4K, NVENC accelerated when available)
- 🖥️ **Gradio UI** – Easy-to-use web interface
- 📂 Outputs: `.txt`, `.lrc`, `.ass`, and `.mp4` video

---

## 🛠️ Installation (Cross-Platform)

> 💡 **Easiest install:** use the [AIraoke-Pinokio launcher](https://github.com/TheAwaken1/AIraoke-Pinokio) for a 1-click install that sets up Python, PyTorch, FFmpeg, and all dependencies automatically.

---

### ⚠️ FFmpeg Setup (if errors occur)

> **Note:** your FFmpeg build must include **libass** (the `subtitles` filter) or videos will render black without lyrics. The builds below include it; some minimal/conda builds do not. AIraoke looks for FFmpeg in the app's `bin/` folder first, then falls back to your system PATH.

If you encounter errors related to **ffmpeg**, download it manually:

1. Go to: [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
2. Download the file: `ffmpeg-release-full.7z`
3. Extract it to a folder (e.g., `C:\ffmpeg7.1.1`)
4. Add bin folder path (e.g., `C:\ffmpeg7.1.1\ffmpeg-7.1.1-full_build\bin`) to your system's environment PATH.
5. Optional: Inside the extracted folder, find and copy:
   - `ffmpeg.exe`
   - `ffplay.exe`
   - `ffprobe.exe`
6. Create a `bin` folder (e.g., `AIraoke\bin`) and paste the above files into it

On Linux/macOS: use your package manager (e.g., `sudo apt install ffmpeg` or `brew install ffmpeg`)


AIraoke works on **Linux**, **Windows**, and **macOS**. Follow the steps below:

---

### ✅ Step 1: Upgrade pip

```bash
pip install --upgrade pip
# or
python.exe -m pip install --upgrade pip
```

---

### ✅ Step 2: Install PyTorch

Choose the appropriate command for your system. For full compatibility, refer to: [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

#### 🖥️ For GPU (NVIDIA CUDA):

**Example for CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Example for CUDA 12.1 or later:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Also tested and working:**
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

#### 🧠 For CPU-only (Linux, Windows, macOS):

```bash
pip install torch torchvision torchaudio
```

---

### ✅ Step 3: Install AIraoke in editable mode

```bash
pip install -e .
```

---

### ✅ Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:

```
demucs
opencv-python
scipy
diffq
pydub
librosa
StrEnum
cattrs
toml
tomli
ffmpeg
ffprobe
bitsandbytes>=0.43.1
transformers>=4.47.0
spacy
```

---

### ✅ Step 5: Additional packages

```bash
pip install gradio==5.29.1 devicetorch
pip install openai-whisper
python -m spacy download en_core_web_sm
```

---

## 🧪 Running the App

```bash
python -m gradio_ui
```

The app will launch at:  
[http://localhost:7860](http://localhost:7860)

---

## 🎤 How to Use

1. **Upload** an audio or video file — or paste a YouTube link (check *Download Video (MP4)* if you want the video as your background). Artist and title auto-fill from the filename.
2. **Transcribe** — pick a Quality Preset (or open Transcription Settings for full control) and click **Transcribe Lyrics**.
3. **Edit & Style** — the app moves you to the Lyrics Editor: fix any words or timing, then open **Video & Audio Style** to choose your background, colors, effects, and vocal volume.
4. **Render** — click **Render Video** and watch the result in the Video Playback tab. Re-render with different styles any time; your transcription is kept.

---

## 📂 Output Files

Transcriptions and videos will be saved in the `output/` directory:

- `.txt` – Original and corrected lyrics
- `.lrc` – Lyric synchronization
- `.ass` – Styled subtitle format
- `.mp4` – Karaoke video 

---

## 🎨 Customization Options

All style options live in **Video & Audio Style** inside the Lyrics Editor tab:

- 🎬 Pick a bundled background, upload a **Custom Video** (any MP4/MOV/WebM — it loops behind your lyrics), or choose **Audio Particles** / Black
- 🎨 Font color, resolution (360p to 4K), video effects, and a background dimmer for text readability
- 🔊 Vocal volume from 100% (original) down to 0% (instrumental only)
- ⏱️ Countdown before the first lyric and beat-synced effects
- ⚙️ Transcription options (GPU, Whisper model, LLM correction) live in **Transcription Settings** on the left

---

## 🙌 Credits

AIraoke is built on top of the amazing work from the open-source community. Special thanks to the following projects and their creators:

- **[Python-Lyric-Transcriber](https://github.com/nomadkaraoke/python-lyrics-transcriber)** by [nomadkaraoke](https://github.com/nomadkaraoke) — The core logic for lyric transcription in AIraoke. We adapted this project by adding a Gradio-based user interface and MP4 video output for visualizing lyric transcriptions.
- **[Whisper](https://github.com/openai/whisper)** by OpenAI — The transcription engine used for converting audio to text.
- **[Demucs](https://github.com/facebookresearch/demucs)** by Meta AI — The vocal separation model used to isolate vocals from audio tracks.
- Gradio UI and enhancements by **TheAwakenOne**

Please respect the licenses and terms of these projects when using AIraoke. Check their respective repositories for details.

---

## 🧠 Built With Love

AIraoke is an experimental tool crafted for creators, karaoke lovers, and audio experimenters. Designed to run locally on PC, Linux, or Mac, it uses Whisper for transcription and an LLM for lyric processing without relying on external APIs. Please note that the output may not always be perfect, as transcription and lyric alignment can vary. I hope AIraoke sparks joy and creativity in your lyric visualization projects, even if it’s a work in progress!

📎 License

AIraoke is licensed under the MIT License. See the LICENSE file for details. This project incorporates code from Python-Lyric-Transcriber, which is also licensed under the MIT License.
