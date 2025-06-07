## ⚠️ Notice: Major App Update

This update introduces improvements to performance, model handling, and the user interface. Please review the changes below:

### 🔧 Model Management

* Models are no longer automatically loaded into GPU memory on app launch.
* You can now **manually load and unload models** as needed, giving you better control over memory usage.

### 🎨 UI Enhancements

* A refreshed user interface has been introduced.
* The app is now **designed for dark mode**. It will still run in light mode, but some visual elements may not display as intended.

### 🐟 Fish Speech Fix

* Fixed a bug where Fish Speech was not chunking text correctly, which could lead to processing issues.

### 📥 Model Download Behavior

* **Chatterbox** and **Kokoro** models will **automatically download** the first time you click "Load" on each one.
* **Fish Speech** models must be **downloaded manually** and are not included in the auto-download process.

### 🗣️ New Feature: Custom Kokoro Voices

* **Kokoro** now supports **custom `.pt` voice models**!
* You can upload your own Kokoro-compatible `.pt` voice files using the **Custom Voice Upload** section in the Kokoro interface.

---
![Screenshot 2025-06-07 204449](https://github.com/user-attachments/assets/d48b4dc0-fca1-47cb-9510-00dcd3d58518)


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
* 🐟 **Fish Speech**: Advanced TTS engine.
* 🎛️ **Professional Audio Effects**: Reverb, Echo, EQ, Pitch shift, Gain.

---

> ## 🚨🚨 **WARNING / IMPORTANT NOTES** 🚨🚨
>
> ⚠️ **Tested Hardware:** This project has **only** been tested on a **Windows 11** machine with an **RTX 4090** GPU.
> 💻 Performance or compatibility on other systems is **not guaranteed**.
>
> 🔊 **Audio Caution:** The **Fish Speech** feature may occasionally produce **extremely loud** or **muffled** audio.
> 🎧 **Please lower your volume and avoid using headphones** during initial tests.

---
## 🛠️ Installation

## Option 1. 
Install via [Pinokio](https://pinokio-home.netlify.app/item?uri=https://github.com/SUP3RMASS1VE/Ultimate-TTS-Studio-SUP3R-Edition-Pinokio).

### Option 2
[Espeak-ng](https://github.com/espeak-ng/espeak-ng) is needed for Kokoro to work at its best.

1. Clone the repository

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

### 4. Install `uv` (optional but recommended for speed)

```bash
pip install uv
```

### 5. Install dependencies

#### Install PyTorch (CUDA 12.8 build)

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### Install other requirements

```bash
uv pip install -r requirements.txt
```

> 💡 If you're not using `uv`, you can just use `pip install` in its place.

---

## 🧠 First-Time Setup Tips

### 📥 Download Fish Speech Model (one-time)

To use **Fish Speech**, you must download the model checkpoint from Hugging Face. This requires a Hugging Face account and access token.

### 🔐 Step-by-Step:

1. **Create an account (if needed):**
   [https://huggingface.co/join](https://huggingface.co/join)

2. **Get your access token:**
   Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a **read token**.

3. **Log in via CLI:**

   ```bash
   huggingface-cli login
   ```

   Paste your token when prompted.

4. **(Optional)** Accept the model license:
   Visit [https://huggingface.co/fishaudio/openaudio-s1-mini](https://huggingface.co/fishaudio/openaudio-s1-mini) and click **"Access repository"** if prompted.

5. **Download the model:**

   ```bash
   huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
   ```

---

Would you like the full updated `README.md` in one code block to copy/paste?


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

## 🙏 Acknowledgments

This project proudly integrates and builds upon the amazing work of:

- [Fish Speech by fishaudio](https://github.com/fishaudio/fish-speech) – Natural and expressive TTS engine.  
  📜 License: [MIT License](https://github.com/fishaudio/fish-speech/blob/main/LICENSE)

- [Kokoro TTS by hexgrad](https://github.com/hexgrad/kokoro) – High-quality multilingual voice synthesis.  
  📜 License: [Apache 2.0 License](https://github.com/hexgrad/kokoro/blob/main/LICENSE)

- [ChatterboxTTS by Resemble AI](https://github.com/resemble-ai/chatterbox) – Custom voice cloning from short reference clips.  
  📜 License: [Apache 2.0 License](https://github.com/resemble-ai/chatterbox/blob/main/LICENSE)

We deeply thank the authors and contributors to these projects for making this work possible.

---

