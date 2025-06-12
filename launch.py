import random
import numpy as np
import torch
import gradio as gr
import os
import subprocess
import sys
import warnings
import re
import json
import io
from datetime import datetime
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
import tempfile
import shutil
from tqdm import tqdm
from scipy.io.wavfile import write

# Chatterbox imports
try:
    from chatterbox.src.chatterbox.tts import ChatterboxTTS
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False
    print("⚠️ ChatterboxTTS not available. Some features will be disabled.")

# Kokoro imports
try:
    from kokoro import KModel, KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    print("⚠️ Kokoro TTS not available. Some features will be disabled.")

# Fish Speech imports
try:
    import queue
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
    from fish_speech.utils.file import audio_to_bytes
    FISH_SPEECH_AVAILABLE = True
except ImportError:
    FISH_SPEECH_AVAILABLE = False
    print("⚠️ Fish Speech not available. Some features will be disabled.")

# eBook Converter imports
try:
    from ebook_converter import (
        EBookConverter, 
        get_supported_formats, 
        analyze_ebook, 
        convert_ebook_to_text_chunks
    )
    EBOOK_CONVERTER_AVAILABLE = True
except ImportError:
    EBOOK_CONVERTER_AVAILABLE = False
    print("⚠️ eBook converter not available. Some features will be disabled.")

# Audio processing imports
try:
    from scipy.signal import butter, filtfilt, hilbert
    from scipy.fft import fft, ifft, fftfreq
    import librosa
    import soundfile as sf
    import base64
    from pydub import AudioSegment
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    print("⚠️ Advanced audio processing libraries not available. Some features will be disabled.")

# ===== HELPER FUNCTIONS =====
def save_audio_with_format(audio_data, sample_rate, output_format="wav", output_folder=None, filename_base=None):
    """
    Save audio data in the specified format (WAV or MP3).
    
    Args:
        audio_data: numpy array of audio samples
        sample_rate: sample rate of the audio
        output_format: "wav" or "mp3"
        output_folder: folder to save the file (default: global output_folder)
        filename_base: base filename without extension (default: auto-generated)
    
    Returns:
        tuple: (filepath, filename) of the saved file
    """
    if output_folder is None:
        output_folder = globals().get('output_folder', 'outputs')
    
    if filename_base is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"tts_output_{timestamp}"
    
    # Ensure output format is lowercase
    output_format = output_format.lower()
    
    # Ensure the folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    if output_format == "wav":
        # Save directly as WAV using scipy
        filename = f"{filename_base}.wav"
        filepath = os.path.join(output_folder, filename)
        write(filepath, sample_rate, (audio_data * 32767).astype(np.int16))
        return filepath, filename
    
    elif output_format == "mp3":
        # Convert to MP3 using pydub
        try:
            # First save as temporary WAV
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            write(temp_wav.name, sample_rate, (audio_data * 32767).astype(np.int16))
            temp_wav.close()
            
            # Convert to MP3
            audio_segment = AudioSegment.from_wav(temp_wav.name)
            filename = f"{filename_base}.mp3"
            filepath = os.path.join(output_folder, filename)
            audio_segment.export(filepath, format="mp3", bitrate="192k")
            
            # Clean up temporary file
            os.unlink(temp_wav.name)
            
            return filepath, filename
            
        except Exception as e:
            print(f"Error converting to MP3: {e}")
            # Fallback to WAV
            print("Falling back to WAV format...")
            filename = f"{filename_base}.wav"
            filepath = os.path.join(output_folder, filename)
            write(filepath, sample_rate, (audio_data * 32767).astype(np.int16))
            return filepath, filename
    
    else:
        raise ValueError(f"Unsupported audio format: {output_format}. Supported formats: wav, mp3")

# ===== GLOBAL CONFIGURATION =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# Cache configuration for Kokoro
cache_base = os.path.abspath(os.path.join(os.getcwd(), 'cache'))
os.environ["HF_HOME"] = os.path.abspath(os.path.join(cache_base, 'HF_HOME'))
os.environ["TORCH_HOME"] = os.path.abspath(os.path.join(cache_base, 'TORCH_HOME'))
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Disable warnings
warnings.filterwarnings("ignore")
torch.nn.utils.parametrize = torch.nn.utils.parametrizations.weight_norm

# ===== DIRECTORY SETUP =====
PRESETS_FILE = "voice_presets.json"
output_folder = os.path.join(os.getcwd(), 'outputs')
custom_voices_folder = os.path.join(os.getcwd(), 'custom_voices')
audiobooks_folder = os.path.join(os.getcwd(), 'audiobooks')

# Create necessary folders
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(custom_voices_folder):
    os.makedirs(custom_voices_folder)

if not os.path.exists(audiobooks_folder):
    os.makedirs(audiobooks_folder)

# ===== MODEL INITIALIZATION =====
CHATTERBOX_MODEL = None
KOKORO_PIPELINES = {}
FISH_SPEECH_ENGINE = None
FISH_SPEECH_LLAMA_QUEUE = None
loaded_voices = {}

# Model loading status
MODEL_STATUS = {
    'chatterbox': {'loaded': False, 'loading': False},
    'kokoro': {'loaded': False, 'loading': False},
    'fish_speech': {'loaded': False, 'loading': False}
}

def init_chatterbox():
    """Initialize ChatterboxTTS model."""
    global CHATTERBOX_MODEL, MODEL_STATUS
    if not CHATTERBOX_AVAILABLE:
        return False, "❌ ChatterboxTTS not available - check installation"
    
    if MODEL_STATUS['chatterbox']['loaded']:
        return True, "✅ ChatterboxTTS already loaded"
    
    if MODEL_STATUS['chatterbox']['loading']:
        return False, "⏳ ChatterboxTTS is currently loading..."
    
    try:
        MODEL_STATUS['chatterbox']['loading'] = True
        print("🔄 Loading ChatterboxTTS...")
        CHATTERBOX_MODEL = ChatterboxTTS.from_pretrained(DEVICE)
        MODEL_STATUS['chatterbox']['loaded'] = True
        MODEL_STATUS['chatterbox']['loading'] = False
        print("✅ ChatterboxTTS loaded successfully")
        return True, "✅ ChatterboxTTS loaded successfully"
    except Exception as e:
        MODEL_STATUS['chatterbox']['loading'] = False
        error_msg = f"❌ Failed to load ChatterboxTTS: {e}"
        print(error_msg)
        return False, error_msg

def unload_chatterbox():
    """Unload ChatterboxTTS model to free memory."""
    global CHATTERBOX_MODEL, MODEL_STATUS
    try:
        if CHATTERBOX_MODEL is not None:
            del CHATTERBOX_MODEL
            CHATTERBOX_MODEL = None
        
        # Force garbage collection
        import gc
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        MODEL_STATUS['chatterbox']['loaded'] = False
        print("✅ ChatterboxTTS unloaded successfully")
        return "✅ ChatterboxTTS unloaded - memory freed"
    except Exception as e:
        error_msg = f"❌ Error unloading ChatterboxTTS: {e}"
        print(error_msg)
        return error_msg

def init_kokoro():
    """Initialize Kokoro TTS models and pipelines."""
    global KOKORO_PIPELINES, MODEL_STATUS
    if not KOKORO_AVAILABLE:
        return False, "❌ Kokoro TTS not available - check installation"
    
    if MODEL_STATUS['kokoro']['loaded']:
        return True, "✅ Kokoro TTS already loaded"
    
    if MODEL_STATUS['kokoro']['loading']:
        return False, "⏳ Kokoro TTS is currently loading..."
    
    try:
        MODEL_STATUS['kokoro']['loading'] = True
        print("🔄 Loading Kokoro TTS...")
        
        # Check if first run
        if not os.path.exists(os.path.join(cache_base, 'HF_HOME/hub/models--hexgrad--Kokoro-82M')):
            print("Downloading/Loading Kokoro models...")
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)
        
        # Load pipelines only (no need for separate KModel)
        KOKORO_PIPELINES = {lang_code: KPipeline(repo_id="hexgrad/Kokoro-82M", lang_code=lang_code) for lang_code in 'abpi'}
        
        # Configure lexicons
        KOKORO_PIPELINES['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
        KOKORO_PIPELINES['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
        
        try:
            if hasattr(KOKORO_PIPELINES['i'].g2p, 'lexicon'):
                KOKORO_PIPELINES['i'].g2p.lexicon.golds['kokoro'] = 'kˈkɔro'
        except Exception as e:
            print(f"Warning: Could not set Italian pronunciation: {e}")
        
        # Re-enable offline mode
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        
        MODEL_STATUS['kokoro']['loaded'] = True
        MODEL_STATUS['kokoro']['loading'] = False
        print("✅ Kokoro TTS loaded successfully")
        return True, "✅ Kokoro TTS loaded successfully"
        
    except Exception as e:
        MODEL_STATUS['kokoro']['loading'] = False
        error_msg = f"❌ Failed to load Kokoro TTS: {e}"
        print(error_msg)
        return False, error_msg

def unload_kokoro():
    """Unload Kokoro TTS models to free memory."""
    global KOKORO_PIPELINES, loaded_voices, MODEL_STATUS
    try:
        # Clear pipelines
        for pipeline in KOKORO_PIPELINES.values():
            del pipeline
        KOKORO_PIPELINES.clear()
        
        # Clear loaded voices
        loaded_voices.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        MODEL_STATUS['kokoro']['loaded'] = False
        print("✅ Kokoro TTS unloaded successfully")
        return "✅ Kokoro TTS unloaded - memory freed"
    except Exception as e:
        error_msg = f"❌ Error unloading Kokoro TTS: {e}"
        print(error_msg)
        return error_msg

def init_fish_speech():
    """Initialize Fish Speech TTS engine."""
    global FISH_SPEECH_ENGINE, FISH_SPEECH_LLAMA_QUEUE, MODEL_STATUS
    if not FISH_SPEECH_AVAILABLE:
        return False, "❌ Fish Speech not available - check installation"
    
    if MODEL_STATUS['fish_speech']['loaded']:
        return True, "✅ Fish Speech already loaded"
    
    if MODEL_STATUS['fish_speech']['loading']:
        return False, "⏳ Fish Speech is currently loading..."
    
    try:
        MODEL_STATUS['fish_speech']['loading'] = True
        print("🔄 Loading Fish Speech...")
        
        # Check for model checkpoints
        checkpoint_path = "checkpoints/openaudio-s1-mini"
        if not os.path.exists(checkpoint_path):
            MODEL_STATUS['fish_speech']['loading'] = False
            error_msg = "❌ Fish Speech checkpoints not found. Please download them first:\nhuggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini"
            print(error_msg)
            return False, error_msg
        
        # Initialize LLAMA queue for text2semantic processing
        precision = torch.half if DEVICE == "cuda" else torch.bfloat16
        FISH_SPEECH_LLAMA_QUEUE = launch_thread_safe_queue(
            checkpoint_path=checkpoint_path,
            device=DEVICE,
            precision=precision,
            compile=False  # Can be enabled for faster inference
        )
        
        # Load decoder model
        decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=os.path.join(checkpoint_path, "codec.pth"),
            device=DEVICE
        )
        
        # Initialize TTS inference engine
        FISH_SPEECH_ENGINE = TTSInferenceEngine(
            llama_queue=FISH_SPEECH_LLAMA_QUEUE,
            decoder_model=decoder_model,
            precision=precision,
            compile=False
        )
        
        MODEL_STATUS['fish_speech']['loaded'] = True
        MODEL_STATUS['fish_speech']['loading'] = False
        print("✅ Fish Speech loaded successfully")
        return True, "✅ Fish Speech loaded successfully"
        
    except Exception as e:
        MODEL_STATUS['fish_speech']['loading'] = False
        error_msg = f"❌ Failed to load Fish Speech: {e}"
        print(error_msg)
        return False, error_msg

def unload_fish_speech():
    """Unload Fish Speech TTS engine to free memory."""
    global FISH_SPEECH_ENGINE, FISH_SPEECH_LLAMA_QUEUE, MODEL_STATUS
    try:
        if FISH_SPEECH_ENGINE is not None:
            del FISH_SPEECH_ENGINE
            FISH_SPEECH_ENGINE = None
        
        if FISH_SPEECH_LLAMA_QUEUE is not None:
            del FISH_SPEECH_LLAMA_QUEUE
            FISH_SPEECH_LLAMA_QUEUE = None
        
        # Force garbage collection
        import gc
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        MODEL_STATUS['fish_speech']['loaded'] = False
        print("✅ Fish Speech unloaded successfully")
        return "✅ Fish Speech unloaded - memory freed"
    except Exception as e:
        error_msg = f"❌ Error unloading Fish Speech: {e}"
        print(error_msg)
        return error_msg

def get_model_status():
    """Get current status of all models."""
    status_text = "📊 **Model Status:**\n\n"
    
    # ChatterboxTTS status
    if CHATTERBOX_AVAILABLE:
        if MODEL_STATUS['chatterbox']['loading']:
            status_text += "🎤 **ChatterboxTTS:** ⏳ Loading...\n"
        elif MODEL_STATUS['chatterbox']['loaded']:
            status_text += "🎤 **ChatterboxTTS:** ✅ Loaded\n"
        else:
            status_text += "🎤 **ChatterboxTTS:** ⭕ Not loaded\n"
    else:
        status_text += "🎤 **ChatterboxTTS:** ❌ Not available\n"
    
    # Kokoro TTS status
    if KOKORO_AVAILABLE:
        if MODEL_STATUS['kokoro']['loading']:
            status_text += "🗣️ **Kokoro TTS:** ⏳ Loading...\n"
        elif MODEL_STATUS['kokoro']['loaded']:
            status_text += "🗣️ **Kokoro TTS:** ✅ Loaded\n"
        else:
            status_text += "🗣️ **Kokoro TTS:** ⭕ Not loaded\n"
    else:
        status_text += "🗣️ **Kokoro TTS:** ❌ Not available\n"
    
    # Fish Speech status
    if FISH_SPEECH_AVAILABLE:
        if MODEL_STATUS['fish_speech']['loading']:
            status_text += "🐟 **Fish Speech:** ⏳ Loading...\n"
        elif MODEL_STATUS['fish_speech']['loaded']:
            status_text += "🐟 **Fish Speech:** ✅ Loaded\n"
        else:
            status_text += "🐟 **Fish Speech:** ⭕ Not loaded\n"
    else:
        status_text += "🐟 **Fish Speech:** ❌ Not available\n"
    
    return status_text

# Don't initialize models at startup - they will be loaded on demand
print("🚀 TTS models ready for on-demand loading...")

# ===== KOKORO VOICE DEFINITIONS =====
KOKORO_CHOICES = {
    '🇺🇸 🚺 Heart ❤️': 'af_heart',
    '🇺🇸 🚺 Bella 🔥': 'af_bella',
    '🇺🇸 🚺 Nicole 🎧': 'af_nicole',
    '🇺🇸 🚺 Aoede': 'af_aoede',
    '🇺🇸 🚺 Kore': 'af_kore',
    '🇺🇸 🚺 Sarah': 'af_sarah',
    '🇺🇸 🚺 Nova': 'af_nova',
    '🇺🇸 🚺 Sky': 'af_sky',
    '🇺🇸 🚺 Alloy': 'af_alloy',
    '🇺🇸 🚺 Jessica': 'af_jessica',
    '🇺🇸 🚺 River': 'af_river',
    '🇺🇸 🚹 Michael': 'am_michael',
    '🇺🇸 🚹 Fenrir': 'am_fenrir',
    '🇺🇸 🚹 Puck': 'am_puck',
    '🇺🇸 🚹 Echo': 'am_echo',
    '🇺🇸 🚹 Eric': 'am_eric',
    '🇺🇸 🚹 Liam': 'am_liam',
    '🇺🇸 🚹 Onyx': 'am_onyx',
    '🇺🇸 🚹 Santa': 'am_santa',
    '🇺🇸 🚹 Adam': 'am_adam',
    '🇬🇧 🚺 Emma': 'bf_emma',
    '🇬🇧 🚺 Isabella': 'bf_isabella',
    '🇬🇧 🚺 Alice': 'bf_alice',
    '🇬🇧 🚺 Lily': 'bf_lily',
    '🇬🇧 🚹 George': 'bm_george',
    '🇬🇧 🚹 Fable': 'bm_fable',
    '🇬🇧 🚹 Lewis': 'bm_lewis',
    '🇬🇧 🚹 Daniel': 'bm_daniel',
    'PF 🚺 Dora': 'pf_dora',
    'PM 🚹 Alex': 'pm_alex',
    'PM 🚹 Santa': 'pm_santa',
    '🇮🇹 🚺 Sara': 'if_sara',
    '🇮🇹 🚹 Nicola': 'im_nicola',
}

# ===== SHARED UTILITY FUNCTIONS =====
def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def split_text_into_chunks(text: str, max_chunk_length: int = 300) -> list[str]:
    """Split text into chunks that respect sentence boundaries."""
    if len(text) <= max_chunk_length:
        return [text]
    
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) + 2 > max_chunk_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if len(sentence) > max_chunk_length:
                    parts = re.split(r'[,;]+', sentence)
                    for part in parts:
                        part = part.strip()
                        if len(current_chunk) + len(part) + 2 > max_chunk_length:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = part
                        else:
                            current_chunk += (", " if current_chunk else "") + part
                else:
                    current_chunk = sentence
        else:
            current_chunk += (". " if current_chunk else "") + sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# ===== AUDIO EFFECTS FUNCTIONS =====
def apply_reverb(audio, sr, room_size=0.3, damping=0.5, wet_level=0.3):
    """Apply reverb effect to audio."""
    if not AUDIO_PROCESSING_AVAILABLE:
        return audio
    
    try:
        reverb_audio = audio.copy()
        delays = [0.01, 0.02, 0.03, 0.05, 0.08]
        gains = [0.6, 0.4, 0.3, 0.2, 0.15]
        
        for delay_time, gain in zip(delays, gains):
            delay_samples = int(sr * delay_time)
            if delay_samples < len(audio):
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples] * gain * (1 - damping)
                reverb_audio += delayed * wet_level
        
        return np.clip(reverb_audio, -1.0, 1.0)
    except Exception as e:
        print(f"Reverb error: {e}")
        return audio

def apply_echo(audio, sr, delay=0.3, decay=0.5):
    """Apply echo effect to audio."""
    try:
        delay_samples = int(sr * delay)
        if delay_samples < len(audio):
            echo_audio = audio.copy()
            echo_audio[delay_samples:] += audio[:-delay_samples] * decay
            return np.clip(echo_audio, -1.0, 1.0)
    except Exception as e:
        print(f"Echo error: {e}")
    return audio

def apply_pitch_shift(audio, sr, semitones):
    """Apply simple pitch shift."""
    try:
        if semitones == 0:
            return audio
        
        factor = 2 ** (semitones / 12.0)
        indices = np.arange(0, len(audio), factor)
        indices = indices[indices < len(audio)].astype(int)
        return audio[indices]
    except Exception as e:
        print(f"Pitch shift error: {e}")
        return audio

def apply_gain(audio, gain_db):
    """Apply gain/volume adjustment in dB."""
    try:
        if gain_db == 0:
            return audio
        
        # Convert dB to linear scale
        gain_linear = 10 ** (gain_db / 20.0)
        gained_audio = audio * gain_linear
        
        # Prevent clipping
        return np.clip(gained_audio, -1.0, 1.0)
    except Exception as e:
        print(f"Gain error: {e}")
        return audio

def apply_eq_filter(audio, sr, freq, gain_db, q_factor=1.0, filter_type='peak'):
    """Apply EQ filter at specific frequency."""
    if not AUDIO_PROCESSING_AVAILABLE:
        return audio
    
    try:
        if gain_db == 0:
            return audio
        
        # Normalize frequency
        nyquist = sr / 2
        norm_freq = freq / nyquist
        
        # Clamp frequency to valid range
        norm_freq = np.clip(norm_freq, 0.01, 0.99)
        
        if filter_type == 'lowpass':
            # Low-pass filter for bass boost
            b, a = butter(2, norm_freq, btype='low')
        elif filter_type == 'highpass':
            # High-pass filter for treble boost
            b, a = butter(2, norm_freq, btype='high')
        elif filter_type == 'peak':
            # Peaking EQ filter (more complex, simplified version)
            # This is a basic implementation - for production use, consider more sophisticated EQ
            b, a = butter(2, [max(0.01, norm_freq - 0.1), min(0.99, norm_freq + 0.1)], btype='band')
        
        # Apply filter
        filtered = filtfilt(b, a, audio)
        
        # Mix with original based on gain
        gain_linear = 10 ** (gain_db / 20.0)
        if gain_db > 0:
            # Boost: blend filtered signal
            mix_ratio = min(gain_linear - 1, 1.0)
            result = audio + filtered * mix_ratio
        else:
            # Cut: reduce filtered signal
            mix_ratio = 1 - (1 / gain_linear)
            result = audio - filtered * mix_ratio
        
        return np.clip(result, -1.0, 1.0)
    except Exception as e:
        print(f"EQ filter error: {e}")
        return audio

def apply_three_band_eq(audio, sr, bass_gain=0, mid_gain=0, treble_gain=0):
    """Apply 3-band EQ (bass, mid, treble)."""
    if not AUDIO_PROCESSING_AVAILABLE:
        return audio
    
    try:
        result = audio.copy()
        
        # Bass: ~80-250 Hz
        if bass_gain != 0:
            result = apply_eq_filter(result, sr, 150, bass_gain, filter_type='lowpass')
        
        # Mid: ~250-4000 Hz  
        if mid_gain != 0:
            result = apply_eq_filter(result, sr, 1000, mid_gain, filter_type='peak')
        
        # Treble: ~4000+ Hz
        if treble_gain != 0:
            result = apply_eq_filter(result, sr, 6000, treble_gain, filter_type='highpass')
        
        return result
    except Exception as e:
        print(f"3-band EQ error: {e}")
        return audio

def apply_audio_effects(audio, sr, effects_settings):
    """Apply selected audio effects to the generated audio."""
    if not effects_settings:
        return audio
    
    processed_audio = audio.copy()
    
    # Apply EQ first (before other effects)
    if effects_settings.get('enable_eq', False):
        processed_audio = apply_three_band_eq(
            processed_audio, sr,
            bass_gain=effects_settings.get('eq_bass', 0),
            mid_gain=effects_settings.get('eq_mid', 0),
            treble_gain=effects_settings.get('eq_treble', 0)
        )
    
    # Apply gain/volume adjustment
    if effects_settings.get('gain_db', 0) != 0:
        processed_audio = apply_gain(
            processed_audio,
            gain_db=effects_settings.get('gain_db', 0)
        )
    
    if effects_settings.get('enable_reverb', False):
        processed_audio = apply_reverb(
            processed_audio, sr,
            room_size=effects_settings.get('reverb_room', 0.3),
            damping=effects_settings.get('reverb_damping', 0.5),
            wet_level=effects_settings.get('reverb_wet', 0.3)
        )
    
    if effects_settings.get('enable_echo', False):
        processed_audio = apply_echo(
            processed_audio, sr,
            delay=effects_settings.get('echo_delay', 0.3),
            decay=effects_settings.get('echo_decay', 0.5)
        )
    
    if effects_settings.get('enable_pitch', False):
        processed_audio = apply_pitch_shift(
            processed_audio, sr,
            semitones=effects_settings.get('pitch_semitones', 0)
        )
    
    return processed_audio

# ===== CHATTERBOX TTS FUNCTIONS =====
def generate_chatterbox_tts(
    text_input: str,
    audio_prompt_path_input: str,
    exaggeration_input: float,
    temperature_input: float,
    seed_num_input: int,
    cfgw_input: float,
    chunk_size_input: int,
    effects_settings=None,
    audio_format: str = "wav",
    skip_file_saving: bool = False
):
    """Generate TTS audio using ChatterboxTTS."""
    if not CHATTERBOX_AVAILABLE:
        return None, "❌ ChatterboxTTS not available - check installation"
    
    if not MODEL_STATUS['chatterbox']['loaded'] or CHATTERBOX_MODEL is None:
        return None, "❌ ChatterboxTTS not loaded - please load the model first"
    
    try:
        if seed_num_input != 0:
            set_seed(int(seed_num_input))
        
        # Split text into chunks
        text_chunks = split_text_into_chunks(text_input, max_chunk_length=chunk_size_input)
        audio_chunks = []
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for chunk in text_chunks:
                wav = CHATTERBOX_MODEL.generate(
                    chunk,
                    audio_prompt_path=audio_prompt_path_input,
                    exaggeration=exaggeration_input,
                    temperature=temperature_input,
                    cfg_weight=cfgw_input,
                )
                audio_chunks.append(wav.squeeze(0).numpy())
        
        # Concatenate chunks
        if len(audio_chunks) == 1:
            final_audio = audio_chunks[0]
        else:
            silence_samples = int(CHATTERBOX_MODEL.sr * 0.05)
            silence = np.zeros(silence_samples)
            
            concatenated_chunks = []
            for i, chunk in enumerate(audio_chunks):
                concatenated_chunks.append(chunk)
                if i < len(audio_chunks) - 1:
                    concatenated_chunks.append(silence)
            
            final_audio = np.concatenate(concatenated_chunks)
        
        # Apply effects
        if effects_settings:
            final_audio = apply_audio_effects(final_audio, CHATTERBOX_MODEL.sr, effects_settings)
        
        # Save audio file in specified format (skip if requested, e.g., for audiobook chunks)
        if skip_file_saving:
            status_message = "✅ Generated with ChatterboxTTS"
        else:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = f"chatterbox_output_{timestamp}"
                filepath, filename = save_audio_with_format(
                    final_audio, CHATTERBOX_MODEL.sr, audio_format, output_folder, filename_base
                )
                status_message = f"✅ Generated with ChatterboxTTS - Saved as: {filename}"
            except Exception as e:
                print(f"Warning: Could not save audio file: {e}")
                status_message = "✅ Generated with ChatterboxTTS (file saving failed)"
        
        return (CHATTERBOX_MODEL.sr, final_audio), status_message
        
    except Exception as e:
        return None, f"❌ ChatterboxTTS error: {str(e)}"

# ===== FISH SPEECH TTS FUNCTIONS =====
def enhance_audio_clarity(audio_data, sample_rate, enhancement_strength=1.0):
    """Enhance audio clarity and reduce muffled sound for Fish Speech."""
    if not AUDIO_PROCESSING_AVAILABLE:
        return audio_data
    
    try:
        from scipy.signal import butter, filtfilt
        
        enhanced_audio = audio_data.copy()
        
        # 1. Apply more aggressive high-frequency emphasis to combat muffled sound
        nyquist = sample_rate / 2
        
        # Multiple frequency bands for better enhancement
        # High-mid boost (2-6 kHz) - critical for speech intelligibility
        high_mid_freq = min(2000, nyquist * 0.4)
        norm_freq = high_mid_freq / nyquist
        b, a = butter(2, norm_freq, btype='high')
        high_mid_content = filtfilt(b, a, audio_data)
        enhanced_audio += high_mid_content * (0.25 * enhancement_strength)  # Adjustable boost for speech clarity
        
        # High frequency boost (4-8 kHz) - for brightness and presence
        high_freq = min(4000, nyquist * 0.6)
        norm_freq = high_freq / nyquist
        b, a = butter(2, norm_freq, btype='high')
        high_freq_content = filtfilt(b, a, audio_data)
        enhanced_audio += high_freq_content * (0.35 * enhancement_strength)  # Adjustable boost for brightness
        
        # 2. Gentle de-emphasis in low-mids to reduce muddiness (200-800 Hz)
        if sample_rate >= 1600:  # Only if sample rate allows
            low_mid_low = max(200, nyquist * 0.02)
            low_mid_high = min(800, nyquist * 0.1)
            norm_low = low_mid_low / nyquist
            norm_high = low_mid_high / nyquist
            
            if norm_high > norm_low and norm_high < 0.95:
                b, a = butter(2, [norm_low, norm_high], btype='band')
                muddy_content = filtfilt(b, a, audio_data)
                enhanced_audio -= muddy_content * 0.15  # Reduce muddiness
        
        # 3. Improved multi-band compression for better dynamics
        # Gentle compression on the whole signal
        threshold_low = 0.15   # Lower threshold for more compression
        ratio_low = 2.5        # Gentler ratio
        
        threshold_high = 0.6   # Higher threshold for peak limiting
        ratio_high = 6.0       # Stronger ratio for peaks
        
        abs_audio = np.abs(enhanced_audio)
        sign = np.sign(enhanced_audio)
        
        # Two-stage compression
        # Stage 1: Gentle compression for overall dynamics
        compressed = np.where(
            abs_audio > threshold_low,
            threshold_low + (abs_audio - threshold_low) / ratio_low,
            abs_audio
        )
        
        # Stage 2: Peak limiting for loud parts
        compressed = np.where(
            compressed > threshold_high,
            threshold_high + (compressed - threshold_high) / ratio_high,
            compressed
        )
        
        enhanced_audio = sign * compressed
        
        # 4. Final normalization with soft knee limiting
        peak = np.max(np.abs(enhanced_audio))
        if peak > 0.85:  # Start limiting earlier
            # Soft knee limiting
            target_peak = 0.85
            enhanced_audio = enhanced_audio * (target_peak / peak)
            
            # Apply soft saturation to remaining peaks
            enhanced_audio = np.tanh(enhanced_audio * 1.2) * 0.85
        
        return enhanced_audio
        
    except Exception as e:
        print(f"Audio enhancement error: {e}")
        return audio_data

def enhance_audio_clarity_minimal(audio_data, sample_rate, enhancement_strength=1.0):
    """Minimal audio enhancement that preserves Fish Speech's natural character."""
    if not AUDIO_PROCESSING_AVAILABLE or enhancement_strength <= 0:
        return audio_data
    
    try:
        from scipy.signal import butter, filtfilt
        
        # Very gentle high-frequency presence boost only (preserve Fish Speech quality)
        enhanced_audio = audio_data.copy()
        
        # Only apply subtle high-frequency emphasis if requested
        if enhancement_strength > 0.5:
            nyquist = sample_rate / 2
            
            # Gentle presence boost around 3-5kHz (speech clarity frequencies)
            presence_freq = min(3500, nyquist * 0.3)
            norm_freq = presence_freq / nyquist
            
            if norm_freq < 0.95:
                b, a = butter(1, norm_freq, btype='high')  # Very gentle 1st order
                presence_content = filtfilt(b, a, audio_data)
                # Much more subtle enhancement
                boost_amount = 0.1 * enhancement_strength  # Max 10% boost
                enhanced_audio += presence_content * boost_amount
        
        # Gentle soft limiting to prevent any artifacts
        peak = np.max(np.abs(enhanced_audio))
        if peak > 0.98:
            enhanced_audio = enhanced_audio * (0.98 / peak)
        
        return enhanced_audio
        
    except Exception as e:
        print(f"Minimal enhancement error: {e}")
        return audio_data

def normalize_audio(audio_data, target_level=-3.0, prevent_clipping=True):
    """Normalize audio to prevent clipping and improve quality - Fish Speech optimized."""
    try:
        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32767.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483647.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Find RMS level for better normalization (more perceptually accurate than peak)
        rms = np.sqrt(np.mean(audio_data ** 2))
        peak = np.max(np.abs(audio_data))
        
        if peak == 0 or rms == 0:
            return audio_data
        
        # Use RMS-based normalization for more natural sound
        target_linear = 10 ** (target_level / 20.0)
        
        # Calculate normalization based on RMS but limited by peak
        rms_factor = target_linear / rms
        peak_factor = 0.9 / peak  # Keep peaks under 0.9
        
        # Use the smaller factor to prevent clipping
        normalization_factor = min(rms_factor, peak_factor)
        
        # Apply normalization
        normalized_audio = audio_data * normalization_factor
        
        # Advanced soft limiting for Fish Speech
        if prevent_clipping:
            # Multi-stage soft limiting
            # Stage 1: Soft knee starting at 0.8
            threshold1 = 0.8
            knee_width = 0.1
            
            abs_audio = np.abs(normalized_audio)
            sign = np.sign(normalized_audio)
            
            # Soft knee compression
            knee_ratio = 3.0
            compressed = np.where(
                abs_audio > threshold1,
                threshold1 + (abs_audio - threshold1) / knee_ratio,
                abs_audio
            )
            
            # Stage 2: Hard limiting at 0.95 with smooth tanh saturation
            threshold2 = 0.95
            final_audio = np.where(
                compressed > threshold2,
                np.tanh((compressed - threshold2) * 2) * 0.05 + threshold2,
                compressed
            )
            
            normalized_audio = sign * final_audio
        
        # Final safety clip (should rarely be needed now)
        normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
        
        return normalized_audio
        
    except Exception as e:
        print(f"Normalization error: {e}")
        return np.clip(audio_data, -1.0, 1.0)

def generate_fish_speech_tts(
    text_input: str,
    fish_ref_audio: str = None,
    fish_ref_text: str = None,
    fish_temperature: float = 0.8,
    fish_top_p: float = 0.8,
    fish_repetition_penalty: float = 1.1,
    fish_max_tokens: int = 1024,
    fish_seed: int = None,
    effects_settings=None,
    audio_format: str = "wav",
    skip_file_saving: bool = False
):
    """Generate TTS audio using Fish Speech - Proper implementation with chunking support."""
    if not FISH_SPEECH_AVAILABLE:
        return None, "❌ Fish Speech not available - check installation"
    
    if not MODEL_STATUS['fish_speech']['loaded'] or FISH_SPEECH_ENGINE is None:
        return None, "❌ Fish Speech not loaded - please load the model first"
    
    try:
        from fish_speech.text import split_text
        
        # Prepare reference audio if provided
        references = []
        if fish_ref_audio and os.path.exists(fish_ref_audio):
            ref_audio_bytes = audio_to_bytes(fish_ref_audio)
            ref_text = fish_ref_text or ""  # Use provided text or empty string
            references.append(ServeReferenceAudio(audio=ref_audio_bytes, text=ref_text))
        
        # Split text into appropriate chunks using Fish Speech's own text splitter
        # This is crucial for handling long texts properly
        chunk_length = 200  # Fish Speech default chunk length for text splitting
        text_chunks = split_text(text_input, chunk_length)
        
        if not text_chunks:
            return None, "❌ No valid text chunks generated"
        
        print(f"Fish Speech - Processing {len(text_chunks)} text chunks")
        for i, chunk in enumerate(text_chunks):
            print(f"  Chunk {i+1}: {chunk[:50]}{'...' if len(chunk) > 50 else ''}")
        
        all_audio_segments = []
        
        # Process each chunk separately
        for i, chunk_text in enumerate(text_chunks):
            print(f"Fish Speech - Processing chunk {i+1}/{len(text_chunks)}")
            
            # Create TTS request for this chunk
            request = ServeTTSRequest(
                text=chunk_text,
                references=references,
                reference_id=None,
                format="wav",
                max_new_tokens=fish_max_tokens,
                chunk_length=chunk_length,  # Internal chunking within Fish Speech
                top_p=fish_top_p,
                repetition_penalty=fish_repetition_penalty,
                temperature=fish_temperature,
                streaming=False,
                use_memory_cache="off",
                seed=fish_seed,
                normalize=True  # Enable text normalization for better stability
            )
            
            # Generate audio for this chunk
            results = list(FISH_SPEECH_ENGINE.inference(request))
            
            # Find the final result for this chunk
            chunk_final_result = None
            for result in results:
                if result.code == "final":
                    chunk_final_result = result
                    break
                elif result.code == "error":
                    return None, f"❌ Fish Speech error in chunk {i+1}: {str(result.error)}"
            
            if chunk_final_result is None or chunk_final_result.error is not None:
                error_msg = str(chunk_final_result.error) if chunk_final_result else f"No audio generated for chunk {i+1}"
                return None, f"❌ Fish Speech error: {error_msg}"
            
            # Extract audio data for this chunk
            sample_rate, chunk_audio_data = chunk_final_result.audio
            
            # Convert to float32
            if chunk_audio_data.dtype != np.float32:
                chunk_audio_data = chunk_audio_data.astype(np.float32)
            
            all_audio_segments.append(chunk_audio_data)
            print(f"Fish Speech - Chunk {i+1} generated: {len(chunk_audio_data)} samples")
        
        # Concatenate all audio segments with small silence between chunks
        if len(all_audio_segments) == 1:
            final_audio = all_audio_segments[0]
        else:
            # Add small silence between chunks (100ms)
            silence_samples = int(sample_rate * 0.1)
            silence = np.zeros(silence_samples, dtype=np.float32)
            
            concatenated_segments = []
            for i, segment in enumerate(all_audio_segments):
                concatenated_segments.append(segment)
                if i < len(all_audio_segments) - 1:  # Don't add silence after last segment
                    concatenated_segments.append(silence)
            
            final_audio = np.concatenate(concatenated_segments)
        
        # Simple safety normalization only if audio is clipped
        peak = np.max(np.abs(final_audio))
        if peak > 1.0:
            final_audio = final_audio / peak
            print(f"Fish Speech - Normalized clipped audio (peak was {peak:.3f})")
        
        print(f"Fish Speech - Final audio: {len(final_audio)} samples, peak: {peak:.3f}")
        
        # Apply user-requested effects
        if effects_settings:
            final_audio = apply_audio_effects(final_audio, sample_rate, effects_settings)
        
        # Save audio file in specified format (skip if requested, e.g., for audiobook chunks)
        if skip_file_saving:
            status_message = f"✅ Generated with Fish Speech ({len(text_chunks)} chunks processed)"
        else:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = f"fish_speech_output_{timestamp}"
                filepath, filename = save_audio_with_format(
                    final_audio, sample_rate, audio_format, output_folder, filename_base
                )
                status_message = f"✅ Generated with Fish Speech ({len(text_chunks)} chunks processed) - Saved as: {filename}"
            except Exception as e:
                print(f"Warning: Could not save audio file: {e}")
                status_message = f"✅ Generated with Fish Speech ({len(text_chunks)} chunks processed) (file saving failed)"
        
        return (sample_rate, final_audio), status_message
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Fish Speech error: {str(e)}"

# ===== KOKORO TTS FUNCTIONS =====
def get_custom_voices():
    """Get custom voices from the custom_voices folder."""
    custom_voices = {}
    custom_voices_folder = os.path.join(os.getcwd(), 'custom_voices')
    if os.path.exists(custom_voices_folder):
        for file in os.listdir(custom_voices_folder):
            file_path = os.path.join(custom_voices_folder, file)
            if file.endswith('.pt') and os.path.isfile(file_path):
                voice_id = os.path.splitext(file)[0]
                custom_voices[f'👤 Custom: {voice_id}'] = f'custom_{voice_id}'
    return custom_voices

def upload_custom_voice(files, voice_name):
    """Upload a custom voice file to the custom_voices folder."""
    if not voice_name or not voice_name.strip():
        return "Please provide a name for your custom voice."
    
    # Sanitize voice name (remove spaces and special characters)
    voice_name = ''.join(c for c in voice_name if c.isalnum() or c == '_')
    
    if not voice_name:
        return "Invalid voice name. Please use alphanumeric characters."
    
    # Check if any files were uploaded
    if not files:
        return "Please upload a .pt voice file."
    
    # In Gradio, the file object structure depends on the file_count parameter
    # For file_count="single", files is the file path as a string
    file_path = files
    
    # Check if the uploaded file is a .pt file
    if not file_path.endswith('.pt'):
        return "Please upload a valid .pt voice file."
    
    # Copy the file to the custom_voices folder with the new name
    target_file = os.path.join(custom_voices_folder, f"{voice_name}.pt")
    
    # If file already exists, remove it
    if os.path.exists(target_file):
        os.remove(target_file)
    
    # Copy the uploaded file
    shutil.copy(file_path, target_file)
    
    # Try to load the voice to verify it works
    voice_id = f'custom_{voice_name}'
    
    try:
        # Load the .pt file directly
        voice_pack = torch.load(target_file, weights_only=True)
        
        # Verify that the voice pack is usable with the model
        # Check if it's a tensor or a list/tuple of tensors
        if not isinstance(voice_pack, (torch.Tensor, list, tuple)):
            raise ValueError("The voice file is not in the expected format (should be a tensor or list of tensors)")
        
        # If it's a list or tuple, check that it contains tensors
        if isinstance(voice_pack, (list, tuple)) and (len(voice_pack) == 0 or not isinstance(voice_pack[0], torch.Tensor)):
            raise ValueError("The voice file does not contain valid tensor data")
            
        loaded_voices[voice_id] = voice_pack
        return f"Custom voice '{voice_name}' uploaded and loaded successfully!"
    except Exception as e:
        # If loading fails, remove the file
        if os.path.exists(target_file):
            os.remove(target_file)
        return f"Error loading custom voice: {str(e)}"

def upload_and_refresh(files, voice_name):
    """Handle custom voice upload and refresh lists."""
    result = upload_custom_voice(files, voice_name)
    
    # If upload was successful, clear the input fields and update voice choices
    if "successfully" in result:
        updated_choices = update_kokoro_voice_choices()
        new_choices = [(k, v) for k, v in updated_choices.items()]
        return result, get_custom_voice_list(), "", None, gr.update(choices=new_choices)
    else:
        return result, get_custom_voice_list(), voice_name, files, gr.update()

def get_custom_voice_list():
    """Get the list of custom voices for the dataframe."""
    # Load any manually added custom voices first
    load_manual_custom_voices()
    
    custom_voices = get_custom_voices()
    if not custom_voices:
        return [["No custom voices found", "N/A"]]
    return [[name.replace('👤 Custom: ', ''), "Loaded"] for name in custom_voices.keys()]

def load_manual_custom_voices():
    """Load custom voices that were manually added to the custom_voices folder."""
    if not os.path.exists(custom_voices_folder):
        return
    
    custom_voices = get_custom_voices()
    for voice_name, voice_id in custom_voices.items():
        # Check if this voice is already loaded
        if voice_id not in loaded_voices:
            try:
                # Extract the actual filename from the voice_id
                voice_filename = voice_id[7:]  # Remove "custom_" prefix (7 characters)
                voice_file = f"{voice_filename}.pt"
                voice_path = os.path.join(custom_voices_folder, voice_file)
                
                if os.path.exists(voice_path):
                    # Load the .pt file directly
                    voice_pack = torch.load(voice_path, weights_only=True)
                    
                    # Verify that the voice pack is usable
                    if isinstance(voice_pack, (torch.Tensor, list, tuple)):
                        loaded_voices[voice_id] = voice_pack
                        print(f"✅ Loaded manually added custom voice: {voice_name}")
                    else:
                        print(f"⚠️ Invalid voice format for {voice_name}")
                else:
                    print(f"⚠️ Voice file not found: {voice_path}")
            except Exception as e:
                print(f"❌ Error loading custom voice {voice_name}: {str(e)}")

def refresh_kokoro_voice_list():
    """Refresh the Kokoro voice list to include new custom voices."""
    # Load any manually added custom voices first
    load_manual_custom_voices()
    
    updated_choices = update_kokoro_voice_choices()
    new_choices = [(k, v) for k, v in updated_choices.items()]
    return gr.update(choices=new_choices)

def update_kokoro_voice_choices():
    """Update choices with custom voices."""
    updated_choices = KOKORO_CHOICES.copy()
    custom_voices = get_custom_voices()
    updated_choices.update(custom_voices)
    return updated_choices

def preload_kokoro_voices():
    """Preload Kokoro voices."""
    if not KOKORO_AVAILABLE or not MODEL_STATUS['kokoro']['loaded']:
        return
    
    print("Preloading Kokoro voices...")
    for voice_name, voice_id in KOKORO_CHOICES.items():
        try:
            pipeline = KOKORO_PIPELINES[voice_id[0]]
            voice_pack = pipeline.load_voice(voice_id)
            loaded_voices[voice_id] = voice_pack
            print(f"Loaded: {voice_name}")
        except Exception as e:
            print(f"Error loading {voice_name}: {e}")
    
    # Load custom voices (both uploaded and manually added)
    load_manual_custom_voices()
    
    print(f"All voices preloaded successfully. Total voices in cache: {len(loaded_voices)}")

def generate_kokoro_tts(text, voice='af_heart', speed=1, effects_settings=None, audio_format="wav", skip_file_saving=False):
    """Generate TTS audio using Kokoro TTS."""
    if not KOKORO_AVAILABLE:
        return None, "❌ Kokoro TTS not available - check installation"
    
    if not MODEL_STATUS['kokoro']['loaded'] or not KOKORO_PIPELINES:
        return None, "❌ Kokoro TTS not loaded - please load the model first"
    
    try:
        # Remove hard character limit and implement chunking instead
        # Split text into chunks (using smaller chunks for Kokoro to maintain quality)
        text_chunks = split_text_into_chunks(text, max_chunk_length=800)  # Kokoro works well with smaller chunks
        audio_chunks = []
        
        # Get voice
        if voice.startswith('custom_'):
            voice_pack = loaded_voices.get(voice)
            if voice_pack is None:
                # Try to load the custom voice if it exists in the folder but isn't cached
                try:
                    # Extract the actual filename from the voice_id
                    # voice format: "custom_filename" -> we want "filename.pt"
                    voice_filename = voice[7:]  # Remove "custom_" prefix (7 characters)
                    voice_file = f"{voice_filename}.pt"
                    voice_path = os.path.join(custom_voices_folder, voice_file)
                    
                    if os.path.exists(voice_path):
                        voice_pack = torch.load(voice_path, weights_only=True)
                        
                        # Verify that the voice pack is usable
                        if isinstance(voice_pack, (torch.Tensor, list, tuple)):
                            loaded_voices[voice] = voice_pack
                            print(f"✅ Auto-loaded custom voice: {voice}")
                        else:
                            return None, f"❌ Invalid voice format for {voice}"
                    else:
                        return None, f"❌ Custom voice file not found: {voice_file}"
                except Exception as e:
                    return None, f"❌ Error loading custom voice {voice}: {str(e)}"
                
            if voice_pack is None:
                return None, f"❌ Custom voice {voice} not found"
            # Use American English pipeline for custom voices
            pipeline = KOKORO_PIPELINES['a']
        else:
            voice_pack = loaded_voices.get(voice)
            if voice_pack is None:
                pipeline = KOKORO_PIPELINES[voice[0]]
                voice_pack = pipeline.load_voice(voice)
                loaded_voices[voice] = voice_pack
            else:
                # Get the correct pipeline for pre-trained voices
                pipeline = KOKORO_PIPELINES[voice[0]]
        
        # Generate audio for each chunk
        for i, chunk in enumerate(text_chunks):
            print(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")
            
            # Use the pipeline as a callable (correct API)
            if voice.startswith('custom_'):
                audio_generator = pipeline(chunk, voice=voice_pack, speed=speed)
            else:
                audio_generator = pipeline(chunk, voice=voice, speed=speed)
            
            # Collect all audio chunks for this text chunk
            chunk_audio_parts = []
            for _, _, audio in audio_generator:
                # Convert tensor to numpy if needed
                if hasattr(audio, 'cpu'):
                    audio = audio.cpu().numpy()
                chunk_audio_parts.append(audio)
            
            # Concatenate parts for this chunk
            if len(chunk_audio_parts) == 1:
                chunk_audio = chunk_audio_parts[0]
            else:
                chunk_audio = np.concatenate(chunk_audio_parts)
            
            audio_chunks.append(chunk_audio)
        
        # Concatenate all chunks with smooth transitions
        if len(audio_chunks) == 1:
            final_audio = audio_chunks[0]
        else:
            # Add small silence between chunks for smooth transitions (shorter than ChatterboxTTS)
            silence_samples = int(24000 * 0.1)  # 100ms silence between chunks
            silence = np.zeros(silence_samples)
            
            concatenated_chunks = []
            for i, chunk in enumerate(audio_chunks):
                concatenated_chunks.append(chunk)
                if i < len(audio_chunks) - 1:  # Don't add silence after last chunk
                    concatenated_chunks.append(silence)
            
            final_audio = np.concatenate(concatenated_chunks)
        
        # Convert tensor to numpy if needed (redundant check, but safe)
        if hasattr(final_audio, 'cpu'):
            final_audio = final_audio.cpu().numpy()
        
        # Apply effects
        if effects_settings:
            final_audio = apply_audio_effects(final_audio, 24000, effects_settings)
        
        # Save audio file in specified format (skip if requested, e.g., for audiobook chunks)
        if skip_file_saving:
            status_message = f"✅ Generated with Kokoro TTS ({len(text_chunks)} chunks)"
        else:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = f"kokoro_output_{timestamp}"
                filepath, filename = save_audio_with_format(
                    final_audio, 24000, audio_format, output_folder, filename_base
                )
                status_message = f"✅ Generated with Kokoro TTS ({len(text_chunks)} chunks) - Saved as: {filename}"
            except Exception as e:
                print(f"Warning: Could not save audio file: {e}")
                status_message = f"✅ Generated with Kokoro TTS ({len(text_chunks)} chunks) (file saving failed)"
        
        return (24000, final_audio), status_message
        
    except Exception as e:
        return None, f"❌ Kokoro error: {str(e)}"

# ===== VOICE PRESET FUNCTIONS =====
def load_voice_presets():
    """Load voice presets from JSON file."""
    try:
        if os.path.exists(PRESETS_FILE):
            with open(PRESETS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading presets: {e}")
    return {}

def save_voice_presets(presets):
    """Save voice presets to JSON file."""
    try:
        with open(PRESETS_FILE, 'w') as f:
            json.dump(presets, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving presets: {e}")
        return False

def save_current_preset(preset_name, tts_engine, **settings):
    """Save current settings as a preset."""
    if not preset_name.strip():
        return "❌ Please enter a preset name", gr.update()
    
    presets = load_voice_presets()
    presets[preset_name.strip()] = {
        'tts_engine': tts_engine,
        'settings': settings,
        'created': datetime.now().isoformat()
    }
    
    if save_voice_presets(presets):
        return f"✅ Preset '{preset_name}' saved!", gr.update(choices=list(presets.keys()))
    else:
        return "❌ Failed to save preset", gr.update()

# ===== EBOOK TO AUDIOBOOK FUNCTIONS =====
def analyze_ebook_file(file_path: str):
    """Analyze an uploaded eBook file and return information."""
    if not EBOOK_CONVERTER_AVAILABLE:
        return {
            'success': False,
            'error': "eBook converter not available. Please install required dependencies.",
            'chapters': [],
            'metadata': None
        }
    
    if not file_path:
        return {
            'success': False,
            'error': "No file uploaded",
            'chapters': [],
            'metadata': None
        }
    
    try:
        info = analyze_ebook(file_path)
        return info
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'chapters': [],
            'metadata': None
        }

def convert_ebook_to_audiobook(
    file_path: str,
    tts_engine: str,
    selected_chapters: list,
    max_chunk_length: int = 500,
    # Audio format parameter
    audio_format: str = "wav",
    # ChatterboxTTS parameters
    chatterbox_ref_audio: str = None,
    chatterbox_exaggeration: float = 0.5,
    chatterbox_temperature: float = 0.8,
    chatterbox_cfg_weight: float = 0.5,
    chatterbox_seed: int = 0,
    # Kokoro parameters
    kokoro_voice: str = 'af_heart',
    kokoro_speed: float = 1.0,
    # Fish Speech parameters
    fish_ref_audio: str = None,
    fish_ref_text: str = None,
    fish_temperature: float = 0.8,
    fish_top_p: float = 0.8,
    fish_repetition_penalty: float = 1.1,
    fish_max_tokens: int = 1024,
    fish_seed: int = None,
    # Effects parameters
    gain_db: float = 0,
    enable_eq: bool = False,
    eq_bass: float = 0,
    eq_mid: float = 0,
    eq_treble: float = 0,
    enable_reverb: bool = False,
    reverb_room: float = 0.3,
    reverb_damping: float = 0.5,
    reverb_wet: float = 0.3,
    enable_echo: bool = False,
    echo_delay: float = 0.3,
    echo_decay: float = 0.5,
    enable_pitch: bool = False,
    pitch_semitones: float = 0,
    # Advanced eBook settings
    chunk_gap: float = 1.0,
    chapter_gap: float = 2.0,
):
    """Convert eBook to audiobook using selected TTS engine."""
    if not EBOOK_CONVERTER_AVAILABLE:
        return None, "❌ eBook converter not available"
    
    if not file_path:
        return None, "❌ No eBook file provided"
    
    try:
        # Convert eBook to text chunks
        text_chunks, metadata = convert_ebook_to_text_chunks(file_path, max_chunk_length)
        
        if not text_chunks:
            return None, "❌ No text content found in eBook"
        
        # Filter chunks based on selected chapters if specified
        if selected_chapters:
            # Convert selected chapter indices to set for faster lookup
            selected_indices = set(selected_chapters)
            text_chunks = [chunk for chunk in text_chunks if chunk['chapter_index'] in selected_indices]
        
        if not text_chunks:
            return None, "❌ No chapters selected for conversion"
        
        # Prepare effects settings
        effects_settings = {
            'gain_db': gain_db,
            'enable_eq': enable_eq,
            'eq_bass': eq_bass,
            'eq_mid': eq_mid,
            'eq_treble': eq_treble,
            'enable_reverb': enable_reverb,
            'reverb_room': reverb_room,
            'reverb_damping': reverb_damping,
            'reverb_wet': reverb_wet,
            'enable_echo': enable_echo,
            'echo_delay': echo_delay,
            'echo_decay': echo_decay,
            'enable_pitch': enable_pitch,
            'pitch_semitones': pitch_semitones,
        } if any([gain_db != 0, enable_eq, enable_reverb, enable_echo, enable_pitch]) else None
        
        # Generate audio for each chunk
        audio_segments = []
        total_chunks = len(text_chunks)
        
        for i, chunk in enumerate(text_chunks):
            print(f"Processing chunk {i+1}/{total_chunks}: {chunk['title']}")
            
            # Generate TTS for this chunk
            if tts_engine == "ChatterboxTTS":
                audio_result, status = generate_chatterbox_tts(
                    chunk['content'], chatterbox_ref_audio, chatterbox_exaggeration,
                    chatterbox_temperature, chatterbox_seed, chatterbox_cfg_weight,
                    max_chunk_length, effects_settings, "wav", skip_file_saving=True  # Skip saving individual chunks
                )
            elif tts_engine == "Kokoro TTS":
                audio_result, status = generate_kokoro_tts(
                    chunk['content'], kokoro_voice, kokoro_speed, effects_settings, "wav", skip_file_saving=True  # Skip saving individual chunks
                )
            elif tts_engine == "Fish Speech":
                audio_result, status = generate_fish_speech_tts(
                    chunk['content'], fish_ref_audio, fish_ref_text, fish_temperature, fish_top_p,
                    fish_repetition_penalty, fish_max_tokens, fish_seed, effects_settings, "wav", skip_file_saving=True  # Skip saving individual chunks
                )
            else:
                return None, f"❌ Invalid TTS engine: {tts_engine}"
            
            if audio_result is None:
                return None, f"❌ Failed to generate audio for chunk {i+1}: {status}"
            
            sample_rate, audio_data = audio_result
            audio_segments.append((sample_rate, audio_data, chunk['title']))
        
        # Concatenate all audio segments
        if not audio_segments:
            return None, "❌ No audio generated"
        
        # Use the sample rate from the first segment
        final_sample_rate = audio_segments[0][0]
        
        # Create silence arrays for different gap types
        chunk_silence_samples = int(final_sample_rate * chunk_gap)
        chapter_silence_samples = int(final_sample_rate * chapter_gap)
        chunk_silence = np.zeros(chunk_silence_samples, dtype=np.float32)
        chapter_silence = np.zeros(chapter_silence_samples, dtype=np.float32)
        
        concatenated_audio = []
        current_chapter_index = None
        
        for i, (sr, audio_data, title) in enumerate(audio_segments):
            # Ensure audio is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            

            
            concatenated_audio.append(audio_data)
            
            # Add appropriate silence between segments (except after the last one)
            if i < len(audio_segments) - 1:
                # Get the chapter index for current and next chunk
                current_chunk_chapter = text_chunks[i]['chapter_index']
                next_chunk_chapter = text_chunks[i + 1]['chapter_index']
                
                # Use chapter gap if moving to a new chapter, otherwise use chunk gap
                if current_chunk_chapter != next_chunk_chapter:
                    concatenated_audio.append(chapter_silence)
                else:
                    concatenated_audio.append(chunk_silence)
        
        final_audio = np.concatenate(concatenated_audio)
        
        # Save the complete audiobook
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        book_title = metadata['title'].replace(' ', '_')
        filename_base = f"audiobook_{book_title}_{timestamp}"
        
        # Normalize and save
        if np.max(np.abs(final_audio)) > 0:
            final_audio = final_audio / np.max(np.abs(final_audio)) * 0.95
        
        # Save in specified format
        try:
            filepath, filename = save_audio_with_format(
                final_audio, final_sample_rate, audio_format, audiobooks_folder, filename_base
            )
        except Exception as e:
            print(f"Warning: Could not save audiobook in {audio_format} format: {e}")
            # Fallback to WAV
            filename = f"{filename_base}.wav"
            filepath = os.path.join(audiobooks_folder, filename)
            write(filepath, final_sample_rate, (final_audio * 32767).astype(np.int16))
        
        # Calculate total duration and file size
        total_duration = len(final_audio) / final_sample_rate / 60  # in minutes
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)  # in MB
        
        status_message = f"✅ Audiobook generated successfully!\n"
        status_message += f"📖 Book: {metadata['title']}\n"
        status_message += f"📊 Chapters processed: {len(audio_segments)}\n"
        status_message += f"⏱️ Total duration: {total_duration:.1f} minutes\n"
        status_message += f"📁 File size: {file_size_mb:.1f} MB\n"
        status_message += f"🔇 Chunk gap: {chunk_gap}s | Chapter gap: {chapter_gap}s\n"
        status_message += f"💾 Saved as: {filename}\n"
        status_message += f"📂 Location: {os.path.abspath(filepath)}\n\n"
        
        # For large files (>50MB or >30 minutes), don't return the audio data to avoid browser issues
        if file_size_mb > 50 or total_duration > 30:
            status_message += "⚠️ Large audiobook detected!\n"
            status_message += "🎧 File too large for browser playback - please use the download link or check the audiobooks folder.\n"
            status_message += "💡 You can play the file with any audio player (VLC, Windows Media Player, etc.)"
            return filepath, status_message  # Return file path instead of audio data
        else:
            status_message += "🎧 Audio preview available below (for smaller files)"
            return (final_sample_rate, final_audio), status_message
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error converting eBook: {str(e)}"

def get_ebook_info_display(analysis_result):
    """Format eBook analysis result for display."""
    if not analysis_result['success']:
        return f"❌ Error: {analysis_result['error']}"
    
    metadata = analysis_result['metadata']
    chapters = analysis_result['chapters']
    
    info_text = f"📖 **{metadata['title']}**\n\n"
    info_text += f"📄 Format: {metadata['format'].upper()}\n"
    info_text += f"📊 File size: {metadata['file_size'] / 1024 / 1024:.1f} MB\n"
    info_text += f"📚 Total chapters: {metadata['total_chapters']}\n"
    info_text += f"📝 Total words: {metadata['total_words']:,}\n"
    info_text += f"⏱️ Estimated duration: {analysis_result['total_estimated_duration']:.1f} minutes\n\n"
    
    info_text += "**📋 Chapters:**\n"
    for i, chapter in enumerate(chapters[:10]):  # Show first 10 chapters
        info_text += f"{i+1}. {chapter['title']} ({chapter['word_count']} words, ~{chapter['estimated_duration']:.1f} min)\n"
    
    if len(chapters) > 10:
        info_text += f"... and {len(chapters) - 10} more chapters\n"
    
    return info_text

# ===== MAIN GENERATION FUNCTION =====
def generate_unified_tts(
    # Common parameters
    text_input: str,
    tts_engine: str,
    # Audio format parameter
    audio_format: str = "wav",
    # ChatterboxTTS parameters
    chatterbox_ref_audio: str = None,
    chatterbox_exaggeration: float = 0.5,
    chatterbox_temperature: float = 0.8,
    chatterbox_cfg_weight: float = 0.5,
    chatterbox_chunk_size: int = 300,
    chatterbox_seed: int = 0,
    # Kokoro parameters
    kokoro_voice: str = 'af_heart',
    kokoro_speed: float = 1.0,
    # Fish Speech parameters
    fish_ref_audio: str = None,
    fish_ref_text: str = None,
    fish_temperature: float = 0.8,
    fish_top_p: float = 0.8,
    fish_repetition_penalty: float = 1.1,
    fish_max_tokens: int = 1024,
    fish_seed: int = None,
    # Effects parameters
    gain_db: float = 0,
    enable_eq: bool = False,
    eq_bass: float = 0,
    eq_mid: float = 0,
    eq_treble: float = 0,
    enable_reverb: bool = False,
    reverb_room: float = 0.3,
    reverb_damping: float = 0.5,
    reverb_wet: float = 0.3,
    enable_echo: bool = False,
    echo_delay: float = 0.3,
    echo_decay: float = 0.5,
    enable_pitch: bool = False,
    pitch_semitones: float = 0,
):
    """Unified TTS generation function."""
    
    if not text_input.strip():
        return None, "❌ Please enter text to synthesize"
    
    # Prepare effects settings
    effects_settings = {
        'gain_db': gain_db,
        'enable_eq': enable_eq,
        'eq_bass': eq_bass,
        'eq_mid': eq_mid,
        'eq_treble': eq_treble,
        'enable_reverb': enable_reverb,
        'reverb_room': reverb_room,
        'reverb_damping': reverb_damping,
        'reverb_wet': reverb_wet,
        'enable_echo': enable_echo,
        'echo_delay': echo_delay,
        'echo_decay': echo_decay,
        'enable_pitch': enable_pitch,
        'pitch_semitones': pitch_semitones,
    } if any([gain_db != 0, enable_eq, enable_reverb, enable_echo, enable_pitch]) else None
    
    if tts_engine == "ChatterboxTTS":
        return generate_chatterbox_tts(
            text_input, chatterbox_ref_audio, chatterbox_exaggeration,
            chatterbox_temperature, chatterbox_seed, chatterbox_cfg_weight,
            chatterbox_chunk_size, effects_settings, audio_format
        )
    elif tts_engine == "Kokoro TTS":
        return generate_kokoro_tts(
            text_input, kokoro_voice, kokoro_speed, effects_settings, audio_format
        )
    elif tts_engine == "Fish Speech":
        return generate_fish_speech_tts(
            text_input, fish_ref_audio, fish_ref_text, fish_temperature, fish_top_p,
            fish_repetition_penalty, fish_max_tokens, fish_seed, effects_settings, audio_format
        )
    else:
        return None, "❌ Invalid TTS engine selected"

# ===== GRADIO INTERFACE =====
def create_gradio_interface():
    """Create the unified Gradio interface."""
    
            # Kokoro voices will be preloaded when the model is loaded
    
    with gr.Blocks(
        title="✨ ULTIMATE TTS STUDIO PRO ✨",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
            neutral_hue="gray",
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
        css="""
        /* CSS Variables for Theme Support */
        :root {
            --text-primary: rgba(255, 255, 255, 0.9);
            --text-secondary: rgba(255, 255, 255, 0.7);
            --text-muted: rgba(255, 255, 255, 0.6);
            --bg-primary: rgba(255, 255, 255, 0.05);
            --bg-secondary: rgba(255, 255, 255, 0.03);
            --border-color: rgba(255, 255, 255, 0.1);
            --accent-color: #667eea;
            --gradient-bg: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        }
        
        /* Light Mode Variables */
        .light :root,
        [data-theme="light"] :root,
        .gradio-container.light {
            --text-primary: rgba(0, 0, 0, 0.9);
            --text-secondary: rgba(0, 0, 0, 0.7);
            --text-muted: rgba(0, 0, 0, 0.6);
            --bg-primary: rgba(0, 0, 0, 0.05);
            --bg-secondary: rgba(0, 0, 0, 0.03);
            --border-color: rgba(0, 0, 0, 0.1);
            --accent-color: #5a67d8;
            --gradient-bg: linear-gradient(135deg, #f7fafc 0%, #edf2f7 50%, #e2e8f0 100%);
        }
        
        /* Auto-detect light mode */
        @media (prefers-color-scheme: light) {
            :root {
                --text-primary: rgba(0, 0, 0, 0.9);
                --text-secondary: rgba(0, 0, 0, 0.7);
                --text-muted: rgba(0, 0, 0, 0.6);
                --bg-primary: rgba(0, 0, 0, 0.05);
                --bg-secondary: rgba(0, 0, 0, 0.03);
                --border-color: rgba(0, 0, 0, 0.1);
                --accent-color: #5a67d8;
                --gradient-bg: linear-gradient(135deg, #f7fafc 0%, #edf2f7 50%, #e2e8f0 100%);
            }
        }
        
        /* Gradio light mode detection */
        .gradio-container[data-theme="light"],
        .gradio-container.light,
        body[data-theme="light"] .gradio-container,
        body.light .gradio-container {
            --text-primary: rgba(0, 0, 0, 0.9) !important;
            --text-secondary: rgba(0, 0, 0, 0.7) !important;
            --text-muted: rgba(0, 0, 0, 0.6) !important;
            --bg-primary: rgba(0, 0, 0, 0.05) !important;
            --bg-secondary: rgba(0, 0, 0, 0.03) !important;
            --border-color: rgba(0, 0, 0, 0.1) !important;
            --accent-color: #5a67d8 !important;
            --gradient-bg: linear-gradient(135deg, #f7fafc 0%, #edf2f7 50%, #e2e8f0 100%) !important;
        }
        
        /* Force light mode styles when body has light class */
        body.light .gradio-container *,
        body[data-theme="light"] .gradio-container *,
        .gradio-container.light *,
        .gradio-container[data-theme="light"] * {
            color: var(--text-primary) !important;
        }
        
        /* Specific overrides for light mode text visibility */
        body.light .gr-markdown,
        body[data-theme="light"] .gr-markdown,
        .gradio-container.light .gr-markdown,
        .gradio-container[data-theme="light"] .gr-markdown {
            color: rgba(0, 0, 0, 0.9) !important;
        }
        
        body.light label,
        body[data-theme="light"] label,
        .gradio-container.light label,
        .gradio-container[data-theme="light"] label {
            color: rgba(0, 0, 0, 0.9) !important;
        }
        
        /* Global Styles */
        .gradio-container {
            max-width: 1800px !important;
            margin: 0 auto !important;
            background: var(--gradient-bg) !important;
            min-height: 100vh;
            font-family: 'Inter', system-ui, sans-serif !important;
        }
        
        /* Animated Background - Responsive */
        .gradio-container::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
            animation: gradientShift 20s ease infinite;
            pointer-events: none;
            z-index: 0;
        }
        
        /* Light mode background adjustment */
        @media (prefers-color-scheme: light) {
            .gradio-container::before {
                background-image: 
                    radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.08) 0%, transparent 50%);
            }
        }
        
        @keyframes gradientShift {
            0%, 100% { transform: rotate(0deg) scale(1); }
            50% { transform: rotate(180deg) scale(1.1); }
        }
        
        /* Main Title Styling - Compact */
        .main-title {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #667eea 100%);
            background-size: 400% 400%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.8em;
            font-weight: 900;
            margin: 15px 0 10px 0;
            text-shadow: 0 0 40px rgba(102, 126, 234, 0.5);
            animation: gradientMove 8s ease infinite;
            letter-spacing: -1px;
            position: relative;
            z-index: 1;
            line-height: 1.1;
        }
        
        @keyframes gradientMove {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .subtitle {
            text-align: center;
            color: var(--text-primary);
            font-size: 1.0em;
            margin-bottom: 20px;
            font-weight: 300;
            letter-spacing: 0.3px;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
            position: relative;
            z-index: 1;
            line-height: 1.3;
        }
        
        /* Glassmorphism Cards - Compact */
        .card, .settings-card, .gr-group {
            background: var(--bg-primary) !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            border-radius: 15px !important;
            padding: 15px !important;
            margin: 8px 0 !important;
            border: 1px solid var(--border-color) !important;
            box-shadow: 
                0 4px 16px 0 rgba(31, 38, 135, 0.25),
                inset 0 0 0 1px var(--border-color) !important;
            transition: all 0.3s ease !important;
            position: relative;
            overflow: hidden;
        }
        
        .card:hover, .settings-card:hover, .gr-group:hover {
            transform: translateY(-5px);
            box-shadow: 
                0 12px 40px 0 rgba(31, 38, 135, 0.5),
                inset 0 0 0 1px var(--border-color) !important;
            background: var(--bg-secondary) !important;
        }
        
        /* Gradient Borders */
        .card::before, .settings-card::before, .gr-group::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            border-radius: 20px;
            padding: 2px;
            background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #4facfe);
            -webkit-mask: 
                linear-gradient(#fff 0 0) content-box, 
                linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .card:hover::before, .settings-card:hover::before, .gr-group:hover::before {
            opacity: 0.5;
        }
        
        /* Generate Button - Compact */
        .generate-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 15px 40px !important;
            font-size: 1.1em !important;
            font-weight: 700 !important;
            color: white !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
            box-shadow: 
                0 6px 20px rgba(102, 126, 234, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            position: relative;
            overflow: hidden;
            margin: 15px auto !important;
            display: block !important;
            width: 300px !important;
        }
        
        .generate-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            transition: left 0.5s ease;
        }
        
        .generate-btn:hover {
            transform: translateY(-3px) scale(1.05) !important;
            box-shadow: 
                0 15px 40px rgba(102, 126, 234, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        }
        
        .generate-btn:hover::before {
            left: 100%;
        }
        
        .generate-btn:active {
            transform: translateY(-1px) scale(1.02) !important;
        }
        
        /* Input Fields */
        .gr-textbox, .gr-dropdown, .gr-slider, .gr-audio, .gr-number {
            background: var(--bg-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            color: var(--text-primary) !important;
            transition: all 0.3s ease !important;
        }
        
        .gr-textbox:focus, .gr-dropdown:focus, .gr-number:focus {
            border-color: var(--accent-color) !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
            background: var(--bg-secondary) !important;
        }
        
        /* Dropdown specific fixes */
        .gr-dropdown {
            position: relative !important;
            z-index: 100 !important;
            isolation: isolate !important;
        }
        
        /* Ensure dropdown container allows overflow and is stable */
        .gr-dropdown > div,
        .gr-dropdown .wrap {
            position: relative !important;
            z-index: 100 !important;
            contain: layout !important;
        }
        
        /* Prevent dropdown from moving during interactions */
        .gr-dropdown,
        .gr-dropdown * {
            backface-visibility: hidden !important;
            transform-style: preserve-3d !important;
        }
        
        /* Dropdown menu styling - Multiple selectors for better compatibility */
        .gr-dropdown .choices,
        .gr-dropdown ul[role="listbox"],
        .gr-dropdown .dropdown-menu,
        .gr-dropdown [role="listbox"],
        .gr-dropdown .svelte-select-list {
            background: var(--bg-primary) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border: 2px solid var(--accent-color) !important;
            border-radius: 12px !important;
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.8),
                0 0 0 1px rgba(102, 126, 234, 0.3) !important;
            z-index: 99999 !important;
            position: absolute !important;
            top: 100% !important;
            left: 0 !important;
            right: 0 !important;
            max-height: 300px !important;
            overflow-y: auto !important;
            margin-top: 4px !important;
            width: 100% !important;
            transform: none !important;
            transition: none !important;
        }
        
        /* Dropdown items */
        .gr-dropdown .choices .item,
        .gr-dropdown li[role="option"],
        .gr-dropdown .dropdown-item,
        .gr-dropdown .svelte-select-list .item {
            color: var(--text-primary) !important;
            padding: 12px 16px !important;
            transition: all 0.2s ease !important;
            background: transparent !important;
            border: none !important;
            cursor: pointer !important;
            white-space: nowrap !important;
            font-size: 0.9em !important;
        }
        
        .gr-dropdown .choices .item:hover,
        .gr-dropdown li[role="option"]:hover,
        .gr-dropdown .dropdown-item:hover,
        .gr-dropdown .svelte-select-list .item:hover {
            background: rgba(102, 126, 234, 0.4) !important;
            color: white !important;
            transform: translateX(2px) !important;
        }
        
        .gr-dropdown .choices .item.selected,
        .gr-dropdown li[role="option"][aria-selected="true"],
        .gr-dropdown .dropdown-item.selected,
        .gr-dropdown .svelte-select-list .item.selected {
            background: rgba(102, 126, 234, 0.6) !important;
            color: white !important;
            font-weight: 600 !important;
        }
        
        /* Fix dropdown container overflow - Apply to all parent containers */
        .gr-group,
        .gr-column,
        .gr-row,
        .gr-accordion,
        .gradio-container {
            overflow: visible !important;
            position: relative !important;
        }
        
        /* Specific fix for accordion content */
        .gr-accordion .gr-accordion-content {
            overflow: visible !important;
        }
        
        /* Prevent parent hover effects from affecting dropdown positioning */
        .gr-group:hover .gr-dropdown,
        .card:hover .gr-dropdown,
        .settings-card:hover .gr-dropdown {
            transform: none !important;
        }
        
        /* Ensure dropdown stays in place during parent transforms */
        .gr-dropdown {
            will-change: auto !important;
        }
        
        /* Override any transform effects on dropdown containers */
        .gr-group:hover,
        .card:hover,
        .settings-card:hover {
            transform: none !important;
        }
        
        /* Ensure dropdown trigger button is properly styled */
        .gr-dropdown button,
        .gr-dropdown .dropdown-toggle {
            background: var(--bg-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            color: var(--text-primary) !important;
            padding: 10px 15px !important;
            width: 100% !important;
            text-align: left !important;
            position: relative !important;
            z-index: 1 !important;
        }
        
        .gr-dropdown button:hover,
        .gr-dropdown .dropdown-toggle:hover {
            border-color: var(--accent-color) !important;
            background: var(--bg-secondary) !important;
        }
        
        /* Arrow icon styling */
        .gr-dropdown button::after,
        .gr-dropdown .dropdown-toggle::after {
            content: '▼' !important;
            float: right !important;
            transition: transform 0.2s ease !important;
        }
        
        .gr-dropdown button[aria-expanded="true"]::after,
        .gr-dropdown .dropdown-toggle.open::after {
            transform: rotate(180deg) !important;
        }
        
        /* Labels */
        label, .gr-label {
            color: var(--text-primary) !important;
            font-weight: 500 !important;
            font-size: 0.95em !important;
            margin-bottom: 8px !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Sliders */
        .gr-slider input[type="range"] {
            background: rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            height: 8px !important;
        }
        
        .gr-slider input[type="range"]::-webkit-slider-thumb {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            border: 2px solid white !important;
            width: 20px !important;
            height: 20px !important;
            border-radius: 50% !important;
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.5) !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }
        
        .gr-slider input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2) !important;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.7) !important;
        }
        
        /* Accordion Styling - Compact */
        .gr-accordion {
            background: var(--bg-secondary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            margin: 12px 0 !important;
        }
        
        .gr-accordion-header {
            background: var(--bg-primary) !important;
            padding: 12px 15px !important;
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            transition: all 0.3s ease !important;
            font-size: 0.95em !important;
        }
        
        .gr-accordion-header:hover {
            background: var(--bg-secondary) !important;
        }
        
        /* Radio Buttons */
        .gr-radio {
            gap: 15px !important;
        }
        
        .gr-radio label {
            background: var(--bg-primary) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 15px 25px !important;
            transition: all 0.3s ease !important;
            cursor: pointer !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .gr-radio label:hover {
            background: var(--bg-secondary) !important;
            border-color: var(--accent-color) !important;
            transform: translateY(-2px) !important;
        }
        
        .gr-radio input[type="radio"]:checked + label {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)) !important;
            border-color: rgba(102, 126, 234, 0.5) !important;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
        }
        
        /* Voice Grid Layout */
        .voice-grid {
            display: grid !important;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) !important;
            gap: 10px !important;
            max-height: 400px !important;
            overflow-y: auto !important;
            padding: 10px !important;
            background: var(--bg-secondary) !important;
            border-radius: 15px !important;
            border: 1px solid var(--border-color) !important;
        }
        
        .voice-grid .gr-radio {
            display: contents !important;
        }
        
        .voice-grid label {
            background: var(--bg-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
            margin: 0 !important;
            font-size: 0.85em !important;
            text-align: center !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            min-height: 40px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        .voice-grid label:hover {
            background: rgba(102, 126, 234, 0.2) !important;
            border-color: var(--accent-color) !important;
            transform: scale(1.02) !important;
        }
        
        /* Multiple selectors to ensure Gradio compatibility */
        .voice-grid input[type="radio"]:checked + label,
        .voice-grid input:checked + label,
        .voice-grid .gr-radio input:checked + label,
        .voice-grid [data-testid="radio"] input:checked + label {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            border: 2px solid #667eea !important;
            color: white !important;
            font-weight: 700 !important;
            box-shadow: 
                0 4px 20px rgba(102, 126, 234, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
            transform: scale(1.05) !important;
            z-index: 10 !important;
            position: relative !important;
        }
        
        /* Ensure selected state persists even on hover */
        .voice-grid input[type="radio"]:checked + label:hover,
        .voice-grid input:checked + label:hover,
        .voice-grid .gr-radio input:checked + label:hover,
        .voice-grid [data-testid="radio"] input:checked + label:hover {
            background: linear-gradient(135deg, #5a67d8, #6b46c1) !important;
            transform: scale(1.05) !important;
            border: 2px solid #5a67d8 !important;
        }
        
        .voice-grid input[type="radio"] {
            display: none !important;
        }
        
        /* Add a checkmark or indicator for selected voice */
        .voice-grid input[type="radio"]:checked + label::after,
        .voice-grid input:checked + label::after,
        .voice-grid .gr-radio input:checked + label::after,
        .voice-grid [data-testid="radio"] input:checked + label::after {
            content: '✓' !important;
            position: absolute !important;
            top: 4px !important;
            right: 6px !important;
            font-size: 14px !important;
            font-weight: bold !important;
            color: white !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8) !important;
        }
        
        /* Ensure the label has relative positioning for the checkmark */
        .voice-grid label {
            position: relative !important;
        }
        
        /* Add a subtle glow animation for selected voice */
        .voice-grid input[type="radio"]:checked + label,
        .voice-grid input:checked + label,
        .voice-grid .gr-radio input:checked + label,
        .voice-grid [data-testid="radio"] input:checked + label {
            animation: selectedGlow 2s ease-in-out infinite alternate !important;
        }
        
        /* Force override any Gradio default styles */
        .voice-grid .gr-radio label[data-selected="true"],
        .voice-grid label[aria-checked="true"],
        .voice-grid label.selected,
        .voice-grid label.voice-selected {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            border: 2px solid #667eea !important;
            color: white !important;
            font-weight: 700 !important;
            transform: scale(1.05) !important;
            box-shadow: 
                0 4px 20px rgba(102, 126, 234, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
            z-index: 10 !important;
            position: relative !important;
        }
        
        /* Checkmark for custom selected class */
        .voice-grid label.voice-selected::after,
        .voice-grid label[data-selected="true"]::after {
            content: '✓' !important;
            position: absolute !important;
            top: 4px !important;
            right: 6px !important;
            font-size: 14px !important;
            font-weight: bold !important;
            color: white !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8) !important;
        }
        
        /* Animation for custom selected class */
        .voice-grid label.voice-selected,
        .voice-grid label[data-selected="true"] {
            animation: selectedGlow 2s ease-in-out infinite alternate !important;
        }
        
        @keyframes selectedGlow {
            0% { 
                box-shadow: 
                    0 4px 20px rgba(102, 126, 234, 0.6),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
            }
            100% { 
                box-shadow: 
                    0 6px 30px rgba(102, 126, 234, 0.8),
                    inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
            }
        }
        
        /* Checkboxes */
        .gr-checkbox {
            background: var(--bg-primary) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        
        .gr-checkbox:checked {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            border-color: transparent !important;
        }
        
        /* Audio Component */
        .gr-audio {
            background: var(--bg-secondary) !important;
            border-radius: 15px !important;
            padding: 20px !important;
        }
        
        /* Section Headers - Compact */
        h2, h3 {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            margin: 10px 0 8px 0 !important;
            position: relative !important;
            padding-left: 12px !important;
            font-size: 1.05em !important;
            line-height: 1.2 !important;
        }
        
        h2::before, h3::before {
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 2px;
            height: 50%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 2px;
        }
        
        /* Settings Group Headers - Smaller and more subtle */
        .gr-group h3, .settings-card h3 {
            font-size: 1.05em !important;
            font-weight: 500 !important;
            margin: 10px 0 8px 0 !important;
            padding-left: 10px !important;
            color: var(--text-primary) !important;
        }
        
        .gr-group h3::before, .settings-card h3::before {
            width: 2px !important;
            height: 50% !important;
        }
        
        /* Info Text */
        .gr-info {
            color: var(--text-muted) !important;
            font-size: 0.85em !important;
            font-style: italic !important;
        }
        
        /* Markdown Styling */
        .gr-markdown {
            color: var(--text-primary) !important;
            line-height: 1.6 !important;
        }
        
        .gr-markdown h3 {
            font-size: 1.05em !important;
            font-weight: 500 !important;
            margin: 8px 0 6px 0 !important;
            padding-left: 8px !important;
            color: var(--text-primary) !important;
        }
        
        .gr-markdown h3::before {
            width: 2px !important;
            height: 45% !important;
        }
        
        .gr-markdown h4 {
            font-size: 0.95em !important;
            font-weight: 500 !important;
            margin: 6px 0 4px 0 !important;
            padding-left: 6px !important;
            color: var(--text-secondary) !important;
        }
        
        .gr-markdown h4::before {
            width: 1.5px !important;
            height: 40% !important;
        }
        
        .gr-markdown strong {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }
        
        .gr-markdown code {
            background: var(--bg-primary) !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-family: 'Fira Code', monospace !important;
        }
        
        /* Status Output */
        .gr-textbox[readonly] {
            background: rgba(0, 255, 0, 0.05) !important;
            border-color: rgba(0, 255, 0, 0.2) !important;
            color: rgba(0, 255, 0, 0.9) !important;
        }
        
        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2, #667eea);
        }
        
        /* Loading Animation */
        .gr-loading {
            color: #667eea !important;
        }
        
        /* Feature Cards - Compact */
        .feature-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            border: 1px solid rgba(102, 126, 234, 0.2);
            border-radius: 12px;
            padding: 12px;
            margin: 5px;
            transition: all 0.3s ease;
            text-align: center;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }
        
        /* Glow Effects */
        .glow {
            box-shadow: 
                0 0 20px rgba(102, 126, 234, 0.5),
                0 0 40px rgba(102, 126, 234, 0.3),
                0 0 60px rgba(102, 126, 234, 0.1);
        }
        
        /* Responsive Design - Compact */
        @media (max-width: 768px) {
            .main-title {
                font-size: 2.2em;
            }
            
            .subtitle {
                font-size: 0.9em;
                margin-bottom: 15px;
            }
            
            .generate-btn {
                width: 100% !important;
                padding: 12px 25px !important;
                font-size: 1.0em !important;
            }
            
            .card, .settings-card, .gr-group {
                padding: 12px !important;
                margin: 5px 0 !important;
            }
            
            .feature-card {
                padding: 8px;
                margin: 3px;
            }
        }
        
        /* Additional light mode fixes */
        .light .main-title,
        [data-theme="light"] .main-title {
            text-shadow: 0 0 40px rgba(102, 126, 234, 0.3) !important;
        }
        
        .light .subtitle,
        [data-theme="light"] .subtitle {
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Dark Theme Overrides */
        .dark {
            --tw-bg-opacity: 0 !important;
        }
        
        /* Custom Animations */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-out;
        }
        """
        + """
        <script>
        // Theme detection and handling
        function updateTheme() {
            const container = document.querySelector('.gradio-container');
            const body = document.body;
            
            // Check for Gradio's theme classes
            if (body.classList.contains('light') || 
                body.hasAttribute('data-theme') && body.getAttribute('data-theme') === 'light' ||
                container && (container.classList.contains('light') || 
                container.hasAttribute('data-theme') && container.getAttribute('data-theme') === 'light')) {
                container.classList.add('light');
                container.setAttribute('data-theme', 'light');
            } else {
                container.classList.remove('light');
                container.removeAttribute('data-theme');
            }
        }
        
        // Run on load
        document.addEventListener('DOMContentLoaded', updateTheme);
        
        // Watch for theme changes
        const observer = new MutationObserver(updateTheme);
        observer.observe(document.body, { 
            attributes: true, 
            attributeFilter: ['class', 'data-theme'],
            subtree: true 
        });
        
        // Also watch for system theme changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', updateTheme);
        }
        
        // Voice selection highlighting fix
        function setupVoiceSelection() {
            const voiceGrids = document.querySelectorAll('.voice-grid');
            voiceGrids.forEach(grid => {
                const radioInputs = grid.querySelectorAll('input[type="radio"]');
                radioInputs.forEach(input => {
                    input.addEventListener('change', function() {
                        // Remove selected class from all labels in this grid
                        const allLabels = grid.querySelectorAll('label');
                        allLabels.forEach(label => {
                            label.classList.remove('voice-selected');
                            label.removeAttribute('data-selected');
                        });
                        
                        // Add selected class to the current label
                        if (this.checked) {
                            const label = this.nextElementSibling;
                            if (label && label.tagName === 'LABEL') {
                                label.classList.add('voice-selected');
                                label.setAttribute('data-selected', 'true');
                            }
                        }
                    });
                });
            });
        }
        
        // Run voice selection setup after DOM loads and when content changes
        document.addEventListener('DOMContentLoaded', setupVoiceSelection);
        
        // Also run when new content is added (Gradio dynamic updates)
        const contentObserver = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length > 0) {
                    setupVoiceSelection();
                }
            });
        });
        
        contentObserver.observe(document.body, {
            childList: true,
            subtree: true
        });
        </script>
        """
    ) as demo:
        
        # Header with enhanced styling
        gr.Markdown("""
        <div class="fade-in">
            <div class="main-title">
            ✨ ULTIMATE TTS STUDIO PRO ✨
            </div>
            <div class="subtitle">
            🎭 ChatterboxTTS + Kokoro TTS + Fish Speech | SUP3R EDITION 🚀<br/>
            <strong>Advanced Text-to-Speech with Multiple Engines, Voice Presets, Audio Effects & Export Options</strong>
            </div>
        </div>
        
        <div style="display: flex; justify-content: center; gap: 10px; margin: 15px 0;">
            <div class="feature-card" style="flex: 1;">
                <h3 style="margin: 0 0 5px 0; padding: 0; font-size: 0.9em;">🎤 Voice Cloning</h3>
                <p style="margin: 0; opacity: 0.8; font-size: 0.8em;">Clone any voice with ChatterboxTTS</p>
            </div>
            <div class="feature-card" style="flex: 1;">
                <h3 style="margin: 0 0 5px 0; padding: 0; font-size: 0.9em;">🗣️ Pre-trained Voices</h3>
                <p style="margin: 0; opacity: 0.8; font-size: 0.8em;">30+ high-quality Kokoro voices</p>
            </div>
            <div class="feature-card" style="flex: 1;">
                <h3 style="margin: 0 0 5px 0; padding: 0; font-size: 0.9em;">📚 eBook Conversion</h3>
                <p style="margin: 0; opacity: 0.8; font-size: 0.8em;">Convert books to audiobooks</p>
            </div>
            <div class="feature-card" style="flex: 1;">
                <h3 style="margin: 0 0 5px 0; padding: 0; font-size: 0.9em;">🎵 Audio Effects</h3>
                <p style="margin: 0; opacity: 0.8; font-size: 0.8em;">Professional audio enhancement</p>
            </div>
        </div>
        """)
        
        # Model Management Section - Compact Version
        with gr.Accordion("🎛️ Model Management", open=True, elem_classes=["fade-in"]):
            gr.Markdown("*Load only the models you need to save memory.*", elem_classes=["fade-in"])
            
            # Compact model status display
            model_status_display = gr.Markdown(
                value=get_model_status(),
                elem_classes=["fade-in"],
                visible=False  # Hide the detailed status by default
            )
            
            with gr.Row():
                # ChatterboxTTS Management - Compact
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("🎤 **ChatterboxTTS**", elem_classes=["fade-in"])
                        chatterbox_status = gr.Markdown(
                            value="⭕ Not loaded" if CHATTERBOX_AVAILABLE else "❌ Not available",
                            elem_classes=["fade-in"]
                        )
                    with gr.Row():
                        load_chatterbox_btn = gr.Button(
                            "🔄 Load",
                            variant="primary",
                            size="sm",
                            visible=CHATTERBOX_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1
                        )
                        unload_chatterbox_btn = gr.Button(
                            "🗑️ Unload",
                            variant="secondary",
                            size="sm",
                            visible=CHATTERBOX_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1
                        )
                
                # Kokoro TTS Management - Compact
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("🗣️ **Kokoro TTS**", elem_classes=["fade-in"])
                        kokoro_status = gr.Markdown(
                            value="⭕ Not loaded" if KOKORO_AVAILABLE else "❌ Not available",
                            elem_classes=["fade-in"]
                        )
                    with gr.Row():
                        load_kokoro_btn = gr.Button(
                            "🔄 Load",
                            variant="primary",
                            size="sm",
                            visible=KOKORO_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1
                        )
                        unload_kokoro_btn = gr.Button(
                            "🗑️ Unload",
                            variant="secondary",
                            size="sm",
                            visible=KOKORO_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1
                        )
                
                # Fish Speech Management - Compact
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("🐟 **Fish Speech**", elem_classes=["fade-in"])
                        fish_status = gr.Markdown(
                            value="⭕ Not loaded" if FISH_SPEECH_AVAILABLE else "❌ Not available",
                            elem_classes=["fade-in"]
                        )
                    with gr.Row():
                        load_fish_btn = gr.Button(
                            "🔄 Load",
                            variant="primary",
                            size="sm",
                            visible=FISH_SPEECH_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1
                        )
                        unload_fish_btn = gr.Button(
                            "🗑️ Unload",
                            variant="secondary",
                            size="sm",
                            visible=FISH_SPEECH_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1
                        )
        
        # Main input section with glassmorphism
        with gr.Row():
            with gr.Column(scale=3):
                # Text input with enhanced styling
                text = gr.Textbox(
                    value="Hello! This is a demonstration of the ULTIMATE TTS STUDIO. You can choose between ChatterboxTTS and Fish Speech for custom voice cloning or Kokoro TTS for high-quality pre-trained voices.",
                    label="📝 Text to synthesize",
                    lines=5,
                    placeholder="Enter your text here...",
                    elem_classes=["fade-in"]
                )
                
                # TTS Engine Selection with custom styling
                tts_engine = gr.Radio(
                    choices=[
                        ("🎤 ChatterboxTTS - Voice Cloning", "ChatterboxTTS"),
                        ("🗣️ Kokoro TTS - Pre-trained Voices", "Kokoro TTS"),
                        ("🐟 Fish Speech - Natural TTS", "Fish Speech")
                    ],
                    value="ChatterboxTTS" if CHATTERBOX_AVAILABLE else "Kokoro TTS" if KOKORO_AVAILABLE else "Fish Speech",
                    label="🎯 Select TTS Engine",
                    info="Choose your preferred text-to-speech engine (auto-selects when you load a model)",
                    elem_classes=["fade-in"]
                )
                
                # Audio Format Selection
                audio_format = gr.Radio(
                    choices=[
                        ("🎵 WAV - Uncompressed (High Quality)", "wav"),
                        ("🎶 MP3 - Compressed (Smaller Size)", "mp3")
                    ],
                    value="wav",
                    label="🎵 Audio Output Format",
                    info="Choose output format: WAV for best quality, MP3 for smaller file size",
                    elem_classes=["fade-in"]
                )
            
            with gr.Column(scale=2):
                # Audio output section with glow effect
                audio_output = gr.Audio(
                    label="🎵 Generated Audio",
                    show_download_button=True,
                    elem_classes=["fade-in", "glow"]
                )
                
                # Status with custom styling
                status_output = gr.Textbox(
                    label="📊 Status",
                    lines=2,
                    interactive=False,
                    elem_classes=["fade-in"]
                )
        
        # Generate button with enhanced animation
        generate_btn = gr.Button(
            "🚀 Generate Speech",
            variant="primary",
            size="lg",
            elem_classes=["generate-btn", "fade-in"]
        )
        
        # Engine-specific settings - All visible at once for easy access
        gr.Markdown("## 🎛️ TTS Engine Settings", elem_classes=["fade-in"])
        gr.Markdown("*Configure settings for all engines below. Only the selected engine will be used for generation.*", elem_classes=["fade-in"])
        
        with gr.Row():
            with gr.Column():
                # ChatterboxTTS Controls
                if CHATTERBOX_AVAILABLE:
                    with gr.Group() as chatterbox_controls:
                        gr.Markdown("**🎤 ChatterboxTTS - Voice cloning from reference audio**")
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                chatterbox_ref_audio = gr.Audio(
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                    label="🎤 Reference Audio File (Optional)",
                                    value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_shadowheart4.flac",
                                    elem_classes=["fade-in"]
                                )
                            
                            with gr.Column(scale=1):
                                chatterbox_exaggeration = gr.Slider(
                                    0.25, 2, step=0.05,
                                    label="🎭 Exaggeration",
                                    value=0.5,
                                    info="Higher = more dramatic",
                                    elem_classes=["fade-in"]
                                )
                                chatterbox_cfg_weight = gr.Slider(
                                    0.2, 1, step=0.05,
                                    label="⚡ CFG Weight",
                                    value=0.5,
                                    info="Speed vs quality",
                                    elem_classes=["fade-in"]
                                )
                        
                        with gr.Accordion("🔧 Advanced ChatterboxTTS Settings", open=False, elem_classes=["fade-in"]):
                            with gr.Row():
                                chatterbox_temperature = gr.Slider(
                                    0.05, 5, step=0.05,
                                    label="🌡️ Temperature",
                                    value=0.8,
                                    info="Higher = more creative"
                                )
                                chatterbox_chunk_size = gr.Slider(
                                    100, 400, step=25,
                                    label="📄 Chunk Size",
                                    value=300,
                                    info="Characters per chunk"
                                )
                                chatterbox_seed = gr.Number(
                                    value=0,
                                    label="🎲 Seed (0=random)",
                                    info="For reproducible results"
                                )
                else:
                    # Placeholder when ChatterboxTTS is not available
                    with gr.Group():
                        gr.Markdown("<div style='text-align: center; padding: 40px; opacity: 0.5;'>**🎤 ChatterboxTTS** - ⚠️ Not available - please check installation</div>")
                        # Create dummy components to maintain consistent interface
                        chatterbox_ref_audio = gr.Audio(visible=False, value=None)
                        chatterbox_exaggeration = gr.Slider(visible=False, value=0.5)
                        chatterbox_temperature = gr.Slider(visible=False, value=0.8)
                        chatterbox_cfg_weight = gr.Slider(visible=False, value=0.5)
                        chatterbox_chunk_size = gr.Slider(visible=False, value=300)
                        chatterbox_seed = gr.Number(visible=False, value=0)
                
                # Kokoro TTS Controls
                if KOKORO_AVAILABLE:
                    with gr.Group() as kokoro_controls:
                        gr.Markdown("**🗣️ Kokoro TTS - High-quality pre-trained voices**")
                        
                        # Voice selection grid
                        gr.Markdown("**🎭 Select Voice**")
                        gr.Markdown("*Choose from pre-trained voices*", elem_classes=["gr-info"])
                        
                        # Create choices as (label, value) pairs
                        kokoro_voice_choices = [(k, v) for k, v in update_kokoro_voice_choices().items()]
                        
                        kokoro_voice = gr.Radio(
                            choices=kokoro_voice_choices,
                            value=list(KOKORO_CHOICES.values())[0] if KOKORO_CHOICES else None,
                            label="",
                            elem_classes=["fade-in", "voice-grid"],
                            show_label=False
                        )
                        
                        # Speed control below the voice grid
                        with gr.Row():
                            kokoro_speed = gr.Slider(
                                0.5, 2.0, step=0.1,
                                label="⚡ Speech Speed",
                                value=1.0,
                                info="Adjust speaking speed",
                                elem_classes=["fade-in"]
                            )
                        
                        # Custom Voice Upload Section
                        with gr.Accordion("👤 Custom Voice Upload", open=False, elem_classes=["fade-in"]):
                            gr.Markdown("""
                            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); 
                                        padding: 12px; border-radius: 12px; margin-bottom: 15px;'>
                                <h3 style='margin: 0 0 5px 0; padding: 0; font-size: 1.0em;'>📁 Upload Your Custom Voices</h3>
                                <p style='margin: 0; opacity: 0.8; font-size: 0.85em;'>Add your own .pt voice files to use with Kokoro TTS</p>
                            </div>
                            """)
                            
                            with gr.Row():
                                with gr.Column(scale=2):
                                    custom_voice_name = gr.Textbox(
                                        label='👤 Custom Voice Name', 
                                        placeholder="Enter a name for your custom voice",
                                        info="Use only letters, numbers, and underscores",
                                        elem_classes=["fade-in"]
                                    )
                                    
                                    custom_voice_files = gr.File(
                                        label="📁 Upload Voice File (.pt)", 
                                        file_count="single",
                                        file_types=[".pt"],
                                        elem_classes=["fade-in"]
                                    )
                                    
                                    with gr.Row():
                                        upload_btn = gr.Button('📤 Upload Voice', variant='primary', elem_classes=["fade-in"])
                                        refresh_voices_btn = gr.Button('🔄 Refresh Voices', variant='secondary', elem_classes=["fade-in"])
                                    
                                    upload_status = gr.Textbox(label="📊 Upload Status", interactive=False, elem_classes=["fade-in"])
                                
                                with gr.Column(scale=1):
                                    gr.Markdown("**📋 Your Custom Voices**")
                                    custom_voice_list = gr.Dataframe(
                                        headers=["Voice Name", "Status"],
                                        datatype=["str", "str"],
                                        row_count=(5, "fixed"),
                                        col_count=(2, "fixed"),
                                        interactive=False,
                                        value=get_custom_voice_list(),
                                        elem_classes=["fade-in"]
                                    )
                            
                            gr.Markdown("""
                            <div style='margin-top: 10px; padding: 10px; background: rgba(102, 126, 234, 0.05); border-radius: 8px; border-left: 3px solid #667eea;'>
                                <p style='margin: 0; font-size: 0.85em; opacity: 0.8;'>
                                    <strong>💡 Tips:</strong> Upload .pt voice files compatible with Kokoro TTS. 
                                    Custom voices will appear with a 👤 prefix in the voice selector above. 
                                    Use the refresh button to update the voice list after uploading.
                                </p>
                            </div>
                            """)
                else:
                    # Placeholder when Kokoro is not available
                    with gr.Group():
                        gr.Markdown("<div style='text-align: center; padding: 40px; opacity: 0.5;'>**🗣️ Kokoro TTS** - ⚠️ Not available - please check installation</div>")
                        # Create dummy components
                        kokoro_voice = gr.Radio(visible=False, value=None, choices=[])
                        kokoro_speed = gr.Slider(visible=False, value=1.0)
                        # Dummy custom voice components
                        custom_voice_name = gr.Textbox(visible=False, value="")
                        custom_voice_files = gr.File(visible=False, value=None)
                        upload_btn = gr.Button(visible=False)
                        refresh_voices_btn = gr.Button(visible=False)
                        upload_status = gr.Textbox(visible=False, value="")
                        custom_voice_list = gr.Dataframe(visible=False, value=[])
            
            with gr.Column():
                # Fish Speech Controls
                if FISH_SPEECH_AVAILABLE:
                    with gr.Group() as fish_speech_controls:
                        gr.Markdown("**🐟 Fish Speech - Natural text-to-speech synthesis**")
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                fish_ref_audio = gr.Audio(
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                    label="🎤 Reference Audio File (Optional)",
                                    value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_shadowheart4.flac",
                                    elem_classes=["fade-in"]
                                )
                            
                            with gr.Column(scale=1):
                                fish_ref_text = gr.Textbox(
                                    label="🗣️ Reference Text (Optional)",
                                    placeholder="Enter reference text here...",
                                    elem_classes=["fade-in"]
                                )
                        
                        with gr.Accordion("🔧 Advanced Fish Speech Settings", open=False, elem_classes=["fade-in"]):
                            gr.Markdown("<p style='opacity: 0.7; margin-bottom: 15px;'>🔧 Fine-tune Fish Speech generation parameters</p>")
                            with gr.Row():
                                fish_temperature = gr.Slider(
                                    0.1, 1.0, step=0.05,
                                    label="🌡️ Temperature",
                                    value=0.8,
                                    info="Higher = more creative (0.1-1.0)"
                                )
                                fish_top_p = gr.Slider(
                                    0.1, 1.0, step=0.05,
                                    label="🎭 Top P",
                                    value=0.8,
                                    info="Controls diversity (0.1-1.0)"
                                )
                                fish_repetition_penalty = gr.Slider(
                                    0.9, 2.0, step=0.05,
                                    label="🔄 Repetition Penalty",
                                    value=1.1,
                                    info="Reduces repetition (0.9-2.0)"
                                )
                            with gr.Row():
                                fish_max_tokens = gr.Slider(
                                    100, 2000, step=100,
                                    label="🔢 Max Tokens",
                                    value=1024,
                                    info="Maximum tokens per chunk"
                                )
                                fish_seed = gr.Number(
                                    value=None,
                                    label="🎲 Seed (None=random)",
                                    info="For reproducible results"
                                )
                            
                            gr.Markdown("### 📝 Text Processing")
                            gr.Markdown("<p style='opacity: 0.7; margin-bottom: 10px;'>Fish Speech automatically splits long texts into chunks for better quality</p>")
                else:
                    # Placeholder when Fish Speech is not available
                    with gr.Group():
                        gr.Markdown("<div style='text-align: center; padding: 40px; opacity: 0.5;'>**🐟 Fish Speech** - ⚠️ Not available - please check installation</div>")
                        # Create dummy components
                        fish_ref_audio = gr.Audio(visible=False, value=None)
                        fish_ref_text = gr.Textbox(visible=False, value="")
                        fish_temperature = gr.Slider(visible=False, value=0.8)
                        fish_top_p = gr.Slider(visible=False, value=0.8)
                        fish_repetition_penalty = gr.Slider(visible=False, value=1.1)
                        fish_max_tokens = gr.Slider(visible=False, value=1024)
                        fish_seed = gr.Number(visible=False, value=None)
        
        # eBook to Audiobook Section
        if EBOOK_CONVERTER_AVAILABLE:
            with gr.Accordion("📚 eBook to Audiobook Converter", open=True, elem_classes=["fade-in"]):
                gr.Markdown("""
                <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); 
                            padding: 15px; border-radius: 12px; margin-bottom: 15px;'>
                    <h3 style='margin: 0 0 8px 0; padding: 0; font-size: 1.1em;'>📖 Convert eBooks to Audiobooks</h3>
                    <p style='margin: 0; opacity: 0.8; font-size: 0.9em;'>
                        Upload your eBook (.epub, .pdf, .txt, .html) and convert it to an audiobook using any TTS engine.
                        .html files work best for automatic chapter detection.
                    </p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # File upload
                        ebook_file = gr.File(
                            label="📁 Upload eBook File",
                            file_types=[".epub", ".pdf", ".txt", ".html", ".htm", ".rtf", ".fb2", ".odt"],
                            elem_classes=["fade-in"]
                        )
                        
                        # Analysis button and results
                        with gr.Row():
                            analyze_btn = gr.Button(
                                "🔍 Analyze eBook",
                                variant="secondary",
                                elem_classes=["fade-in"]
                            )
                            convert_ebook_btn = gr.Button(
                                "🎧 Convert to Audiobook",
                                variant="primary",
                                elem_classes=["fade-in"]
                            )
                            clear_ebook_btn = gr.Button(
                                "🗑️ Clear",
                                variant="secondary",
                                elem_classes=["fade-in"]
                            )
                        
                        # eBook information display
                        ebook_info = gr.Markdown(
                            value="Upload an eBook file and click 'Analyze eBook' to see details.",
                            elem_classes=["fade-in"]
                        )
                        
                        # Chapter selection
                        chapter_selection = gr.CheckboxGroup(
                            label="📋 Select Chapters to Convert (leave empty for all)",
                            choices=[],
                            value=[],
                            visible=False,
                            elem_classes=["fade-in"]
                        )
                    
                    with gr.Column(scale=1):
                        # Conversion settings
                        gr.Markdown("**⚙️ Conversion Settings**")
                        
                        ebook_tts_engine = gr.Radio(
                            choices=[
                                ("🎤 ChatterboxTTS", "ChatterboxTTS"),
                                ("🗣️ Kokoro TTS", "Kokoro TTS"),
                                ("🐟 Fish Speech", "Fish Speech")
                            ],
                            value="ChatterboxTTS" if CHATTERBOX_AVAILABLE else "Kokoro TTS" if KOKORO_AVAILABLE else "Fish Speech",
                            label="🎯 TTS Engine for Audiobook",
                            elem_classes=["fade-in"]
                        )
                        
                        # Audio Format for eBook conversion
                        ebook_audio_format = gr.Radio(
                            choices=[
                                ("🎵 WAV - Uncompressed (High Quality)", "wav"),
                                ("🎶 MP3 - Compressed (Smaller Size)", "mp3")
                            ],
                            value="wav",
                            label="🎵 Audiobook Format",
                            info="Choose format: WAV for best quality, MP3 for smaller file size",
                            elem_classes=["fade-in"]
                        )
                        
                        ebook_chunk_length = gr.Slider(
                            300, 800, step=50,
                            label="📄 Text Chunk Length",
                            value=500,
                            info="Characters per TTS chunk",
                            elem_classes=["fade-in"]
                        )
                        
                        # Chunk timing controls for eBook conversion
                        with gr.Accordion("⏱️ Chunk Timing Controls", open=True, elem_classes=["fade-in"]):
                            gr.Markdown("""
                            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); 
                                        padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
                                <p style='margin: 0; opacity: 0.8; font-size: 0.85em;'>
                                    🔇 Control the silence duration between chunks and chapters in your audiobook
                                </p>
                            </div>
                            """)
                            
                            ebook_chunk_gap = gr.Slider(
                                0.0, 3.0, step=0.1,
                                label="🔇 Gap Between Chunks (seconds)",
                                value=1.0,
                                info="Silence duration between text chunks within the same chapter",
                                elem_classes=["fade-in"]
                            )
                            
                            ebook_chapter_gap = gr.Slider(
                                0.0, 5.0, step=0.1,
                                label="📖 Gap Between Chapters (seconds)",
                                value=2.0,
                                info="Silence duration when transitioning between chapters",
                                elem_classes=["fade-in"]
                            )
                        
                        # Audiobook output
                        audiobook_output = gr.Audio(
                            label="🎧 Generated Audiobook",
                            show_download_button=True,
                            elem_classes=["fade-in", "glow"]
                        )
                        
                        # Download link for large files
                        audiobook_download = gr.File(
                            label="📥 Download Large Audiobook",
                            visible=False,
                            elem_classes=["fade-in"]
                        )
                        
                        # Conversion status
                        ebook_status = gr.Textbox(
                            label="📊 Conversion Status",
                            lines=4,
                            interactive=False,
                            elem_classes=["fade-in"]
                        )
                
                # Supported formats info
                supported_formats = get_supported_formats() if EBOOK_CONVERTER_AVAILABLE else {}
                gr.Markdown(f"""
                <div style='margin-top: 15px; padding: 12px; background: rgba(102, 126, 234, 0.05); border-radius: 8px; border-left: 3px solid #667eea;'>
                    <p style='margin: 0; font-size: 0.85em; opacity: 0.8;'>
                        <strong>📋 Supported Formats:</strong> {', '.join(supported_formats.keys()) if supported_formats else 'N/A'}<br/>
                        <strong>💡 Best Results:</strong> .html files work best for automatic chapter detection.<br/>
                        <strong>⚡ Performance:</strong> Large books may take several minutes to convert depending on length and TTS engine.<br/>
                        <strong>📁 Large Files:</strong> Audiobooks >50MB or >30min will be saved to the audiobooks folder with a download link (browser can't play very large files).<br/>
                        <strong>🎧 Playback:</strong> Use VLC, Windows Media Player, or any audio player for large audiobooks.
                    </p>
                </div>
                """)
        else:
            # Placeholder when eBook converter is not available
            with gr.Accordion("📚 eBook to Audiobook Converter", open=False, elem_classes=["fade-in"]):
                gr.Markdown("""
                <div style='text-align: center; padding: 40px; opacity: 0.5;'>
                    <h3>📚 eBook to Audiobook Converter</h3>
                    <p>⚠️ Not available - please install required dependencies:</p>
                    <code>pip install ebooklib PyPDF2 beautifulsoup4 chardet</code>
                </div>
                """)
        
        # Audio Effects in a separate expandable section
        with gr.Accordion("🎵 Audio Effects Studio", open=False, elem_classes=["fade-in"]):
            gr.Markdown("""
            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); 
                        padding: 12px; border-radius: 12px; margin-bottom: 15px;'>
                <h3 style='margin: 0 0 5px 0; padding: 0; font-size: 1.0em;'>🎚️ Professional Audio Processing</h3>
                <p style='margin: 0; opacity: 0.8; font-size: 0.85em;'>Add studio-quality effects to enhance your generated speech</p>
            </div>
            """)
            
            # Volume and EQ Section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🔊 Volume & EQ Settings")
                    gain_db = gr.Slider(-20, 20, step=0.5, label="🎚️ Master Gain (dB)", value=0, 
                                       info="Boost or reduce overall volume", elem_classes=["fade-in"])
                    
                    enable_eq = gr.Checkbox(label="Enable 3-Band EQ", value=False, elem_classes=["fade-in"])
                    with gr.Row():
                        eq_bass = gr.Slider(-12, 12, step=0.5, label="🔈 Bass", value=0, 
                                          info="80-250 Hz", elem_classes=["fade-in"])
                        eq_mid = gr.Slider(-12, 12, step=0.5, label="🔉 Mid", value=0, 
                                         info="250-4000 Hz", elem_classes=["fade-in"])
                        eq_treble = gr.Slider(-12, 12, step=0.5, label="🔊 Treble", value=0, 
                                            info="4000+ Hz", elem_classes=["fade-in"])
            
            # Effects Section with better layout
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🏛️ Spatial Effects")
                    enable_reverb = gr.Checkbox(label="Enable Reverb", value=False, elem_classes=["fade-in"])
                    with gr.Column():
                        reverb_room = gr.Slider(0.1, 1.0, step=0.1, label="Room Size", value=0.3, elem_classes=["fade-in"])
                        reverb_damping = gr.Slider(0.1, 1.0, step=0.1, label="Damping", value=0.5, elem_classes=["fade-in"])
                        reverb_wet = gr.Slider(0.1, 0.8, step=0.1, label="Wet Mix", value=0.3, elem_classes=["fade-in"])
                
                with gr.Column():
                    gr.Markdown("#### 🔊 Time-Based Effects")
                    enable_echo = gr.Checkbox(label="Enable Echo", value=False, elem_classes=["fade-in"])
                    with gr.Column():
                        echo_delay = gr.Slider(0.1, 1.0, step=0.1, label="Delay Time (s)", value=0.3, elem_classes=["fade-in"])
                        echo_decay = gr.Slider(0.1, 0.9, step=0.1, label="Decay Amount", value=0.5, elem_classes=["fade-in"])
                
                with gr.Column():
                    gr.Markdown("#### 🎼 Pitch Effects")
                    enable_pitch = gr.Checkbox(label="Enable Pitch Shift", value=False, elem_classes=["fade-in"])
                    pitch_semitones = gr.Slider(-12, 12, step=1, label="Pitch (semitones)", value=0, 
                                               info="±12 semitones = ±1 octave", elem_classes=["fade-in"])
        

        
        # Footer with credits - Compact
        gr.Markdown("""
        <div style='text-align: center; margin-top: 20px; padding: 15px; 
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05)); 
                    border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.1);'>
            <p style='opacity: 0.7; margin: 0; font-size: 0.85em;'>
                Made with ❤️ by SUP3RMASS1VE | 
                <a href='https://github.com/SUP3RMASS1VE/Ultimate-TTS-Studio-SUP3R-Edition-Pinokio' target='_blank' style='color: #667eea; text-decoration: none;'>GitHub</a> | 
                <a href='https://discord.gg/mvDcrA57AQ' target='_blank' style='color: #667eea; text-decoration: none;'>Discord</a>
            </p>
        </div>
        """)
        
        # Model management event handlers - Updated for compact interface with auto-selection
        def handle_load_chatterbox():
            success, message = init_chatterbox()
            if success:
                chatterbox_status_text = "✅ Loaded (Auto-selected)"
                # Auto-select ChatterboxTTS engine when loaded
                selected_engine = "ChatterboxTTS"
            else:
                chatterbox_status_text = "❌ Failed to load"
                selected_engine = gr.update()  # No change to current selection
            
            if EBOOK_CONVERTER_AVAILABLE:
                return chatterbox_status_text, selected_engine, selected_engine
            else:
                return chatterbox_status_text, selected_engine
        
        def handle_unload_chatterbox():
            message = unload_chatterbox()
            chatterbox_status_text = "⭕ Not loaded"
            # Don't change engine selection when unloading
            return chatterbox_status_text
        
        def handle_load_kokoro():
            success, message = init_kokoro()
            if success:
                preload_kokoro_voices()  # Preload voices after loading model
                kokoro_status_text = "✅ Loaded (Auto-selected)"
                # Auto-select Kokoro TTS engine when loaded
                selected_engine = "Kokoro TTS"
            else:
                kokoro_status_text = "❌ Failed to load"
                selected_engine = gr.update()  # No change to current selection
            
            if EBOOK_CONVERTER_AVAILABLE:
                return kokoro_status_text, selected_engine, selected_engine
            else:
                return kokoro_status_text, selected_engine
        
        def handle_unload_kokoro():
            message = unload_kokoro()
            kokoro_status_text = "⭕ Not loaded"
            # Don't change engine selection when unloading
            return kokoro_status_text
        
        def handle_load_fish():
            success, message = init_fish_speech()
            if success:
                fish_status_text = "✅ Loaded (Auto-selected)"
                # Auto-select Fish Speech engine when loaded
                selected_engine = "Fish Speech"
            else:
                fish_status_text = "❌ Failed to load"
                selected_engine = gr.update()  # No change to current selection
            
            if EBOOK_CONVERTER_AVAILABLE:
                return fish_status_text, selected_engine, selected_engine
            else:
                return fish_status_text, selected_engine
        
        def handle_unload_fish():
            message = unload_fish_speech()
            fish_status_text = "⭕ Not loaded"
            # Don't change engine selection when unloading
            return fish_status_text
        
        # ChatterboxTTS management
        if CHATTERBOX_AVAILABLE:
            load_chatterbox_btn.click(
                fn=handle_load_chatterbox,
                outputs=[chatterbox_status, tts_engine, ebook_tts_engine] if EBOOK_CONVERTER_AVAILABLE else [chatterbox_status, tts_engine]
            )
            unload_chatterbox_btn.click(
                fn=handle_unload_chatterbox,
                outputs=[chatterbox_status]
            )
        
        # Kokoro TTS management
        if KOKORO_AVAILABLE:
            load_kokoro_btn.click(
                fn=handle_load_kokoro,
                outputs=[kokoro_status, tts_engine, ebook_tts_engine] if EBOOK_CONVERTER_AVAILABLE else [kokoro_status, tts_engine]
            )
            unload_kokoro_btn.click(
                fn=handle_unload_kokoro,
                outputs=[kokoro_status]
            )
        
        # Fish Speech management
        if FISH_SPEECH_AVAILABLE:
            load_fish_btn.click(
                fn=handle_load_fish,
                outputs=[fish_status, tts_engine, ebook_tts_engine] if EBOOK_CONVERTER_AVAILABLE else [fish_status, tts_engine]
            )
            unload_fish_btn.click(
                fn=handle_unload_fish,
                outputs=[fish_status]
            )
        
        # Main generation event handler
        generate_btn.click(
            fn=generate_unified_tts,
            inputs=[
                text, tts_engine, audio_format,
                chatterbox_ref_audio, chatterbox_exaggeration, chatterbox_temperature,
                chatterbox_cfg_weight, chatterbox_chunk_size, chatterbox_seed,
                kokoro_voice, kokoro_speed,
                fish_ref_audio, fish_ref_text, fish_temperature, fish_top_p, fish_repetition_penalty, fish_max_tokens, fish_seed,
                gain_db, enable_eq, eq_bass, eq_mid, eq_treble,
                enable_reverb, reverb_room, reverb_damping, reverb_wet,
                enable_echo, echo_delay, echo_decay,
                enable_pitch, pitch_semitones
            ],
            outputs=[audio_output, status_output]
        )
        
        # eBook conversion event handlers
        if EBOOK_CONVERTER_AVAILABLE:
            def handle_ebook_analysis(file_path):
                """Handle eBook file analysis."""
                if not file_path:
                    return "Please upload an eBook file first.", gr.update(choices=[], visible=False)
                
                analysis_result = analyze_ebook_file(file_path)
                info_display = get_ebook_info_display(analysis_result)
                
                if analysis_result['success']:
                    # Create chapter choices for selection
                    chapter_choices = [
                        (f"{i+1}. {ch['title']}", i) 
                        for i, ch in enumerate(analysis_result['chapters'])
                    ]
                    return info_display, gr.update(choices=chapter_choices, visible=True)
                else:
                    return info_display, gr.update(choices=[], visible=False)
            
            def handle_ebook_conversion(
                file_path, tts_engine_choice, selected_chapters, chunk_length, ebook_format,
                # All the TTS parameters need to be passed through
                cb_ref_audio, cb_exag, cb_temp, cb_cfg, cb_seed,
                kok_voice, kok_speed,
                fish_ref_audio, fish_ref_text, fish_temp, fish_top_p, fish_rep_pen, fish_max_tok, fish_seed_val,
                gain, eq_en, eq_b, eq_m, eq_t,
                rev_en, rev_room, rev_damp, rev_wet,
                echo_en, echo_del, echo_dec,
                pitch_en, pitch_semi,
                # Advanced eBook settings
                chunk_gap, chapter_gap
            ):
                """Handle eBook to audiobook conversion."""
                if not file_path:
                    return None, None, "Please upload an eBook file first."
                
                result = convert_ebook_to_audiobook(
                    file_path, tts_engine_choice, selected_chapters, chunk_length, ebook_format,
                    cb_ref_audio, cb_exag, cb_temp, cb_cfg, cb_seed,
                    kok_voice, kok_speed,
                    fish_ref_audio, fish_ref_text, fish_temp, fish_top_p, fish_rep_pen, fish_max_tok, fish_seed_val,
                    gain, eq_en, eq_b, eq_m, eq_t,
                    rev_en, rev_room, rev_damp, rev_wet,
                    echo_en, echo_del, echo_dec,
                    pitch_en, pitch_semi,
                    # Advanced eBook settings
                    chunk_gap, chapter_gap
                )
                
                if result[0] is None:
                    # Error case
                    return None, gr.update(visible=False), result[1]
                
                audio_result, status_message = result
                
                # Check if result is a file path (large file) or audio data (small file)
                if isinstance(audio_result, str):
                    # Large file - return file path for download
                    return None, gr.update(value=audio_result, visible=True), status_message
                else:
                    # Small file - return audio data for playback
                    return audio_result, gr.update(visible=False), status_message
            
            def handle_clear_ebook():
                """Clear all eBook-related inputs and outputs."""
                return (
                    None,  # ebook_file
                    "Upload an eBook file and click 'Analyze eBook' to see details.",  # ebook_info
                    gr.update(choices=[], value=[], visible=False),  # chapter_selection
                    None,  # audiobook_output
                    gr.update(visible=False),  # audiobook_download
                    ""     # ebook_status
                )
            
            # Connect eBook analysis
            analyze_btn.click(
                fn=handle_ebook_analysis,
                inputs=[ebook_file],
                outputs=[ebook_info, chapter_selection]
            )
            
            # Connect eBook conversion
            convert_ebook_btn.click(
                fn=handle_ebook_conversion,
                inputs=[
                    ebook_file, ebook_tts_engine, chapter_selection, ebook_chunk_length, ebook_audio_format,
                    # ChatterboxTTS parameters
                    chatterbox_ref_audio, chatterbox_exaggeration, chatterbox_temperature,
                    chatterbox_cfg_weight, chatterbox_seed,
                    # Kokoro parameters
                    kokoro_voice, kokoro_speed,
                    # Fish Speech parameters
                    fish_ref_audio, fish_ref_text, fish_temperature, fish_top_p, 
                    fish_repetition_penalty, fish_max_tokens, fish_seed,
                    # Effects parameters
                    gain_db, enable_eq, eq_bass, eq_mid, eq_treble,
                    enable_reverb, reverb_room, reverb_damping, reverb_wet,
                    enable_echo, echo_delay, echo_decay,
                    enable_pitch, pitch_semitones,
                    # Advanced eBook settings
                    ebook_chunk_gap, ebook_chapter_gap
                ],
                outputs=[audiobook_output, audiobook_download, ebook_status]
            )
            
            # Connect eBook clear button
            clear_ebook_btn.click(
                fn=handle_clear_ebook,
                inputs=[],
                outputs=[ebook_file, ebook_info, chapter_selection, audiobook_output, audiobook_download, ebook_status]
            )
        
        # Custom voice upload event handlers (only if Kokoro is available)
        if KOKORO_AVAILABLE:
            # Upload custom voice
            upload_btn.click(
                fn=upload_and_refresh,
                inputs=[custom_voice_files, custom_voice_name],
                outputs=[upload_status, custom_voice_list, custom_voice_name, custom_voice_files, kokoro_voice]
            )
            
            # Refresh voice list
            refresh_voices_btn.click(
                fn=refresh_kokoro_voice_list,
                outputs=[kokoro_voice]
            )
            
            # Refresh custom voice list
            refresh_voices_btn.click(
                fn=get_custom_voice_list,
                outputs=[custom_voice_list]
            )
    
    return demo

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("🚀 Starting Unified TTS Pro...")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    ) 
