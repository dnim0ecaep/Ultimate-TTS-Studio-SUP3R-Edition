
---

# ✨ Ultimate TTS Studio SUP3R Edition ✨

**Ultimate TTS Studio** is a powerful all-in-one text-to-speech studio that brings together **ChatterboxTTS**, **Kokoro TTS**, and **Fish Speech** under one interactive Gradio interface.

🎭 Reference Audio Cloning
🗣️ Pre-trained Multi-Language Voices
🐟 Natural TTS with Audio Effects
🎵 Real-time Voice Synthesis & Export

---

## 🚀 Features

* 🎤 **ChatterboxTTS**: Custom voice cloning using short reference clips.
* 🗣️ **Kokoro TTS**: High-quality, multilingual pre-trained voices.
* 🐟 **Fish Speech**: Advanced TTS engine with clarity enhancement.
* 🎛️ **Professional Audio Effects**: Reverb, Echo, EQ, Pitch shift, Gain.
* 💾 **Voice Presets**: Save and reuse voice configurations.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/SUP3RMASS1VE/Ultimate-TTS-Studio-SUP3R-Edition.git
cd Ultimate-TTS-Studio-SUP3R-Edition
```

### 2. Create a Python virtual environment

```bash
python -m venv env
```

### 3. Activate the environment

* **Windows**:

  ```bash
  env\Scripts\activate
  ```

* **macOS/Linux**:

  ```bash
  source env/bin/activate
  ```

### 4. Install `uv` (optional but recommended for speed)

```bash
pip install uv
```

### 5. Install dependencies

#### Install PyTorch (CUDA 12.4 build)

```bash
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

#### Install other requirements

```bash
uv pip install -r requirements.txt
```

> 💡 If you're not using `uv`, you can just use `pip install` in its place.

---

## 🧠 First-Time Setup Tips

* To use **Fish Speech**, download the model checkpoint:

  ```bash
  huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
  ```

* Kokoro models are automatically downloaded on first run.

---

## ▶️ Run the Studio

```bash
python launch.py
```

This will launch a local Gradio interface at:
📍 `http://127.0.0.1:7860`

---

## 💡 Notes

* All engines are optional — the app will gracefully disable missing engines.
* ChatterboxTTS and Fish Speech support reference audio input.
* Audio effects are applied post-synthesis for professional-quality output.
* Custom Kokoro voices can be added to `custom_voices/` as `.pt` files.

---

## 📜 License

MIT License © SUP3RMASS1VE

---

Would you like this saved to a file or customized with usage screenshots, Docker support, or Hugging Face Spaces deploy instructions?
