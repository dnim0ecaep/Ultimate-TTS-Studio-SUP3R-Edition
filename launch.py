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

# Audio processing imports
try:
    from scipy.signal import butter, filtfilt, hilbert
    from scipy.fft import fft, ifft, fftfreq
    import librosa
    import soundfile as sf
    import base64
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    print("⚠️ Advanced audio processing libraries not available. Some features will be disabled.")

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
PRESETS_AUDIO_DIR = "saved_voices"
EXPORT_DIR = "exports"
output_folder = os.path.join(os.getcwd(), 'outputs')
custom_voices_folder = os.path.join(os.getcwd(), 'custom_voices')

for directory in [PRESETS_AUDIO_DIR, EXPORT_DIR, output_folder, custom_voices_folder]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# ===== MODEL INITIALIZATION =====
CHATTERBOX_MODEL = None
KOKORO_PIPELINES = {}
FISH_SPEECH_ENGINE = None
FISH_SPEECH_LLAMA_QUEUE = None
loaded_voices = {}

def init_chatterbox():
    """Initialize ChatterboxTTS model."""
    global CHATTERBOX_MODEL
    if not CHATTERBOX_AVAILABLE:
        return False
    
    try:
        print("🔄 Loading ChatterboxTTS...")
        CHATTERBOX_MODEL = ChatterboxTTS.from_pretrained(DEVICE)
        print("✅ ChatterboxTTS loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load ChatterboxTTS: {e}")
        return False

def init_kokoro():
    """Initialize Kokoro TTS models and pipelines."""
    global KOKORO_PIPELINES
    if not KOKORO_AVAILABLE:
        return False
    
    try:
        print("🔄 Loading Kokoro TTS...")
        
        # Check if first run
        if not os.path.exists(os.path.join(cache_base, 'HF_HOME/hub/models--hexgrad--Kokoro-82M')):
            print("First run detected, downloading Kokoro models...")
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
        
        print("✅ Kokoro TTS loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load Kokoro TTS: {e}")
        return False

def init_fish_speech():
    """Initialize Fish Speech TTS engine."""
    global FISH_SPEECH_ENGINE, FISH_SPEECH_LLAMA_QUEUE
    if not FISH_SPEECH_AVAILABLE:
        return False
    
    try:
        print("🔄 Loading Fish Speech...")
        
        # Check for model checkpoints
        checkpoint_path = "checkpoints/openaudio-s1-mini"
        if not os.path.exists(checkpoint_path):
            print("Fish Speech checkpoints not found. Please download them first:")
            print("huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini")
            return False
        
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
        
        print("✅ Fish Speech loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load Fish Speech: {e}")
        return False

# Initialize models at startup
print("🚀 Initializing TTS models...")
chatterbox_available = init_chatterbox()
kokoro_available = init_kokoro()
fish_speech_available = init_fish_speech()

if not chatterbox_available and not kokoro_available and not fish_speech_available:
    print("❌ No TTS models available. Please check your installation.")

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
    effects_settings=None
):
    """Generate TTS audio using ChatterboxTTS."""
    if not chatterbox_available or CHATTERBOX_MODEL is None:
        return None, "❌ ChatterboxTTS not available"
    
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
        
        return (CHATTERBOX_MODEL.sr, final_audio), "✅ Generated with ChatterboxTTS"
        
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
    effects_settings=None
):
    """Generate TTS audio using Fish Speech - Clean implementation following official Fish Speech patterns."""
    if not FISH_SPEECH_AVAILABLE or FISH_SPEECH_ENGINE is None:
        return None, "❌ Fish Speech not available"
    
    try:
        # Prepare reference audio if provided
        references = []
        if fish_ref_audio and os.path.exists(fish_ref_audio):
            ref_audio_bytes = audio_to_bytes(fish_ref_audio)
            ref_text = fish_ref_text or ""  # Use provided text or empty string
            references.append(ServeReferenceAudio(audio=ref_audio_bytes, text=ref_text))
        
        # Create TTS request using official Fish Speech defaults (similar to their webui)
        request = ServeTTSRequest(
            text=text_input,
            references=references,
            reference_id=None,
            format="wav",
            max_new_tokens=fish_max_tokens,
            chunk_length=300,  # Use chunking like official implementation
            top_p=fish_top_p,  # Don't limit, let user control
            repetition_penalty=fish_repetition_penalty,
            temperature=fish_temperature,  # Don't limit, let user control
            streaming=False,
            use_memory_cache="off",
            seed=fish_seed
        )
        
        # Generate audio using the same pattern as official Fish Speech webui
        results = list(FISH_SPEECH_ENGINE.inference(request))
        
        # Find the final result
        final_result = None
        for result in results:
            if result.code == "final":
                final_result = result
                break
            elif result.code == "error":
                return None, f"❌ Fish Speech error: {str(result.error)}"
        
        if final_result is None or final_result.error is not None:
            error_msg = str(final_result.error) if final_result else "No audio generated"
            return None, f"❌ Fish Speech error: {error_msg}"
        
        # Extract audio data - use as-is from Fish Speech (this is the key!)
        sample_rate, audio_data = final_result.audio
        
        # Convert to float32 and ensure proper range (Fish Speech should output proper levels)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Simple safety normalization only if audio is clipped
        peak = np.max(np.abs(audio_data))
        if peak > 1.0:
            audio_data = audio_data / peak  # Only normalize if actually clipped
            print(f"Fish Speech - Normalized clipped audio (peak was {peak:.3f})")
        
        print(f"Fish Speech - Raw output peak: {peak:.3f} ({20*np.log10(peak + 1e-10):.1f} dB)")
        
        # Apply user-requested effects only (post-processing like official Fish Speech)
        if effects_settings:
            audio_data = apply_audio_effects(audio_data, sample_rate, effects_settings)
        
        final_peak = np.max(np.abs(audio_data))
        print(f"Fish Speech - Final peak: {final_peak:.3f}")
        
        return (sample_rate, audio_data), "✅ Generated with Fish Speech (clean output)"
        
    except Exception as e:
        return None, f"❌ Fish Speech error: {str(e)}"

# ===== KOKORO TTS FUNCTIONS =====
def get_custom_voices():
    """Get custom voices from the custom_voices folder."""
    custom_voices = {}
    if os.path.exists(custom_voices_folder):
        for file in os.listdir(custom_voices_folder):
            file_path = os.path.join(custom_voices_folder, file)
            if file.endswith('.pt') and os.path.isfile(file_path):
                voice_id = os.path.splitext(file)[0]
                custom_voices[f'👤 Custom: {voice_id}'] = f'custom_{voice_id}'
    return custom_voices

def update_kokoro_voice_choices():
    """Update choices with custom voices."""
    updated_choices = KOKORO_CHOICES.copy()
    custom_voices = get_custom_voices()
    updated_choices.update(custom_voices)
    return updated_choices

def preload_kokoro_voices():
    """Preload Kokoro voices."""
    if not kokoro_available:
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
    
    # Load custom voices
    custom_voices = get_custom_voices()
    for voice_name, voice_id in custom_voices.items():
        try:
            voice_file = f"{voice_id.split('_')[1]}.pt"
            voice_path = os.path.join(custom_voices_folder, voice_file)
            
            if os.path.exists(voice_path):
                voice_pack = torch.load(voice_path, weights_only=True)
                loaded_voices[voice_id] = voice_pack
                print(f"Loaded custom: {voice_name}")
        except Exception as e:
            print(f"Error loading custom {voice_name}: {e}")

def generate_kokoro_tts(text, voice='af_heart', speed=1, effects_settings=None):
    """Generate TTS audio using Kokoro TTS."""
    if not kokoro_available:
        return None, "❌ Kokoro TTS not available"
    
    try:
        # Remove hard character limit and implement chunking instead
        # Split text into chunks (using smaller chunks for Kokoro to maintain quality)
        text_chunks = split_text_into_chunks(text, max_chunk_length=800)  # Kokoro works well with smaller chunks
        audio_chunks = []
        
        # Get voice
        if voice.startswith('custom_'):
            voice_pack = loaded_voices.get(voice)
            if voice_pack is None:
                return None, f"❌ Custom voice {voice} not found"
        else:
            voice_pack = loaded_voices.get(voice)
            if voice_pack is None:
                pipeline = KOKORO_PIPELINES[voice[0]]
                voice_pack = pipeline.load_voice(voice)
                loaded_voices[voice] = voice_pack
        
        # Generate audio for each chunk
        pipeline = KOKORO_PIPELINES[voice[0]]
        
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
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kokoro_output_{timestamp}.wav"
        filepath = os.path.join(output_folder, filename)
        
        write(filepath, 24000, (final_audio * 32767).astype(np.int16))
        
        return (24000, final_audio), f"✅ Generated with Kokoro TTS ({len(text_chunks)} chunks): {filepath}"
        
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

# ===== MAIN GENERATION FUNCTION =====
def generate_unified_tts(
    # Common parameters
    text_input: str,
    tts_engine: str,
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
            chatterbox_chunk_size, effects_settings
        )
    elif tts_engine == "Kokoro TTS":
        return generate_kokoro_tts(
            text_input, kokoro_voice, kokoro_speed, effects_settings
        )
    elif tts_engine == "Fish Speech":
        return generate_fish_speech_tts(
            text_input, fish_ref_audio, fish_ref_text, fish_temperature, fish_top_p,
            fish_repetition_penalty, fish_max_tokens, fish_seed, effects_settings
        )
    else:
        return None, "❌ Invalid TTS engine selected"

# ===== GRADIO INTERFACE =====
def create_gradio_interface():
    """Create the unified Gradio interface."""
    
    # Preload Kokoro voices if available
    if kokoro_available:
        preload_kokoro_voices()
    
    with gr.Blocks(
        title="✨ ULTIMATE TTS STUDIO PRO ✨",
        css="""
        .gradio-container {
            max-width: 1600px !important;
            margin: 0 auto !important;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .generate-btn {
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            border: none;
            border-radius: 12px;
            padding: 15px 30px;
            font-size: 18px;
            font-weight: bold;
            color: white;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        .generate-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .settings-card {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
            padding: 15px;
            margin: 5px 0;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .main-title {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3em;
            font-weight: 900;
            margin: 20px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            text-align: center;
            color: #888;
            font-size: 1.2em;
            margin-bottom: 30px;
            font-weight: 300;
        }
        """
    ) as demo:
        
        # Header
        gr.Markdown("""
        <div class="main-title">
        ✨ ULTIMATE TTS STUDIO PRO ✨
        </div>
        <div class="subtitle">
        🎭 ChatterboxTTS + Kokoro TTS + Fish Speech | SUP3R EDITION 🚀<br/>
        <strong>Advanced Text-to-Speech with Multiple Engines, Voice Presets, Audio Effects & Export Options</strong>
        </div>
        
        <div style="text-align: center; margin: 20px 0; padding: 15px; background: linear-gradient(90deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1)); border-radius: 10px; border: 1px solid rgba(102,126,234,0.2);">
        🔥 <strong>Choose between ChatterboxTTS (reference audio cloning) and Kokoro TTS (pre-trained voices) for your text-to-speech needs!</strong> 🔥
        </div>
        """)
        
        # Main input section
        with gr.Row():
            with gr.Column(scale=3):
                # Text input
                text = gr.Textbox(
                    value="Hello! This is a demonstration of the ULTIMATE TTS STUDIO PRO. You can choose between ChatterboxTTS for custom voice cloning or Kokoro TTS for high-quality pre-trained voices.",
                    label="📝 Text to synthesize",
                    lines=4,
                    placeholder="Enter your text here..."
                )
                
                # TTS Engine Selection
                tts_engine = gr.Radio(
                    choices=[
                        ("🎤 ChatterboxTTS (Reference Audio Cloning)", "ChatterboxTTS"),
                        ("🗣️ Kokoro TTS (Pre-trained Voices)", "Kokoro TTS"),
                        ("🐟 Fish Speech (Text-to-Speech)", "Fish Speech")
                    ],
                    value="ChatterboxTTS" if chatterbox_available else "Kokoro TTS" if kokoro_available else "Fish Speech",
                    label="🎯 Select TTS Engine",
                    info="Choose your preferred text-to-speech engine"
                )
            
            with gr.Column(scale=2):
                # Audio output section
                audio_output = gr.Audio(
                    label="🎵 Generated Audio",
                    show_download_button=True
                )
                
                # Status
                status_output = gr.Textbox(
                    label="📊 Status",
                    lines=2,
                    interactive=False
                )
        
        # Generate button - single prominent button
        generate_btn = gr.Button(
            "🚀 Generate Speech",
            variant="primary",
            size="lg",
            elem_classes=["generate-btn"]
        )
        
        # Engine-specific settings - All visible at once for easy access
        gr.Markdown("## 🎛️ TTS Engine Settings")
        gr.Markdown("*Configure settings for all engines below. Only the selected engine will be used for generation.*")
        
        with gr.Row():
            with gr.Column():
                # ChatterboxTTS Controls
                if chatterbox_available:
                    with gr.Group() as chatterbox_controls:
                        gr.Markdown("### 🎤 ChatterboxTTS Settings")
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                chatterbox_ref_audio = gr.Audio(
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                    label="🎤 Reference Audio File (Optional)",
                                    value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_shadowheart4.flac"
                                )
                            
                            with gr.Column(scale=1):
                                chatterbox_exaggeration = gr.Slider(
                                    0.25, 2, step=0.05,
                                    label="🎭 Exaggeration",
                                    value=0.5,
                                    info="Higher = more dramatic"
                                )
                                chatterbox_cfg_weight = gr.Slider(
                                    0.2, 1, step=0.05,
                                    label="⚡ CFG Weight",
                                    value=0.5,
                                    info="Speed vs quality"
                                )
                        
                        with gr.Accordion("🔧 Advanced ChatterboxTTS Settings", open=False):
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
                        gr.Markdown("### 🎤 ChatterboxTTS Settings")
                        gr.Markdown("*ChatterboxTTS not available - please check installation*")
                        # Create dummy components to maintain consistent interface
                        chatterbox_ref_audio = gr.Audio(visible=False, value=None)
                        chatterbox_exaggeration = gr.Slider(visible=False, value=0.5)
                        chatterbox_temperature = gr.Slider(visible=False, value=0.8)
                        chatterbox_cfg_weight = gr.Slider(visible=False, value=0.5)
                        chatterbox_chunk_size = gr.Slider(visible=False, value=300)
                        chatterbox_seed = gr.Number(visible=False, value=0)
                
                # Kokoro TTS Controls
                if kokoro_available:
                    with gr.Group() as kokoro_controls:
                        gr.Markdown("### 🗣️ Kokoro TTS Settings")
                        
                        with gr.Row():
                            with gr.Column():
                                # Create choices as (label, value) pairs
                                kokoro_voice_choices = [(k, v) for k, v in update_kokoro_voice_choices().items()]
                                
                                kokoro_voice = gr.Dropdown(
                                    choices=kokoro_voice_choices,
                                    value=list(KOKORO_CHOICES.values())[0] if KOKORO_CHOICES else None,
                                    label="🎭 Select Voice",
                                    info="Choose from pre-trained voices"
                                )
                            
                            with gr.Column():
                                kokoro_speed = gr.Slider(
                                    0.5, 2.0, step=0.1,
                                    label="⚡ Speech Speed",
                                    value=1.0,
                                    info="Adjust speaking speed"
                                )
                else:
                    # Placeholder when Kokoro is not available
                    with gr.Group():
                        gr.Markdown("### 🗣️ Kokoro TTS Settings")
                        gr.Markdown("*Kokoro TTS not available - please check installation*")
                        # Create dummy components
                        kokoro_voice = gr.Dropdown(visible=False, value=None)
                        kokoro_speed = gr.Slider(visible=False, value=1.0)
            
            with gr.Column():
                # Fish Speech Controls
                if fish_speech_available:
                    with gr.Group() as fish_speech_controls:
                        gr.Markdown("### 🐟 Fish Speech Settings")
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                fish_ref_audio = gr.Audio(
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                    label="🎤 Reference Audio File (Optional)",
                                    value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_shadowheart4.flac"
                                )
                            
                            with gr.Column(scale=1):
                                fish_ref_text = gr.Textbox(
                                    label="🗣️ Reference Text (Optional)",
                                    placeholder="Enter reference text here..."
                                )
                        
                        with gr.Accordion("🔧 Advanced Fish Speech Settings", open=False):
                            with gr.Row():
                                fish_temperature = gr.Slider(
                                    0.05, 5, step=0.05,
                                    label="🌡️ Temperature",
                                    value=0.8,
                                    info="Higher = more creative"
                                )
                                fish_top_p = gr.Slider(
                                    0.1, 1.0, step=0.05,
                                    label="🎭 Top P",
                                    value=0.8,
                                    info="Controls diversity of generated text"
                                )
                                fish_repetition_penalty = gr.Slider(
                                    0.1, 2.0, step=0.05,
                                    label="🔄 Repetition Penalty",
                                    value=1.1,
                                    info="Controls the likelihood of repeating tokens"
                                )
                            with gr.Row():
                                fish_max_tokens = gr.Slider(
                                    100, 2000, step=100,
                                    label="🔢 Max Tokens",
                                    value=1024,
                                    info="Maximum number of tokens to generate"
                                )
                                fish_seed = gr.Number(
                                    value=None,
                                    label="🎲 Seed (None=random)",
                                    info="For reproducible results"
                                )
                            

                else:
                    # Placeholder when Fish Speech is not available
                    with gr.Group():
                        gr.Markdown("### 🐟 Fish Speech Settings")
                        gr.Markdown("*Fish Speech not available - please check installation*")
                        # Create dummy components
                        fish_ref_audio = gr.Audio(visible=False, value=None)
                        fish_ref_text = gr.Textbox(visible=False, value="")
                        fish_temperature = gr.Slider(visible=False, value=0.8)
                        fish_top_p = gr.Slider(visible=False, value=0.8)
                        fish_repetition_penalty = gr.Slider(visible=False, value=1.1)
                        fish_max_tokens = gr.Slider(visible=False, value=1024)
                        fish_seed = gr.Number(visible=False, value=None)
        
        # Audio Effects in a separate expandable section
        with gr.Accordion("🎵 Audio Effects - Professional Enhancement", open=True):
            gr.Markdown("### Add professional audio effects to enhance your generated speech")
            
            # Volume and EQ Section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🔊 Volume & EQ Settings")
                    gain_db = gr.Slider(-20, 20, step=0.5, label="🎚️ Gain/Volume (dB)", value=0, info="Boost or reduce overall volume")
                    
                    enable_eq = gr.Checkbox(label="Enable 3-Band EQ", value=False)
                    with gr.Row():
                        eq_bass = gr.Slider(-12, 12, step=0.5, label="🔈 Bass (dB)", value=0, info="Low frequencies (80-250 Hz)")
                        eq_mid = gr.Slider(-12, 12, step=0.5, label="🔉 Mid (dB)", value=0, info="Mid frequencies (250-4000 Hz)")
                        eq_treble = gr.Slider(-12, 12, step=0.5, label="🔊 Treble (dB)", value=0, info="High frequencies (4000+ Hz)")
            
            # Effects Section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🏛️ Reverb Settings")
                    enable_reverb = gr.Checkbox(label="Enable Reverb", value=False)
                    with gr.Row():
                        reverb_room = gr.Slider(0.1, 1.0, step=0.1, label="Room Size", value=0.3)
                        reverb_damping = gr.Slider(0.1, 1.0, step=0.1, label="Damping", value=0.5)
                        reverb_wet = gr.Slider(0.1, 0.8, step=0.1, label="Reverb Amount", value=0.3)
                
                with gr.Column():
                    gr.Markdown("#### 🔊 Echo Settings")
                    enable_echo = gr.Checkbox(label="Enable Echo", value=False)
                    with gr.Row():
                        echo_delay = gr.Slider(0.1, 1.0, step=0.1, label="Echo Delay (s)", value=0.3)
                        echo_decay = gr.Slider(0.1, 0.9, step=0.1, label="Echo Decay", value=0.5)
                
                with gr.Column():
                    gr.Markdown("#### 🎼 Pitch Settings")
                    enable_pitch = gr.Checkbox(label="Enable Pitch Shift", value=False)
                    pitch_semitones = gr.Slider(-12, 12, step=1, label="Pitch (semitones)", value=0, info="Positive = higher, Negative = lower")
        
        # Tips section
        with gr.Accordion("💡 Tips & Usage Guide", open=False):
            gr.Markdown("""
            ### 🎯 Choosing the Right Engine
            
            **ChatterboxTTS** - Best for:
            - Custom voice cloning from reference audio
            - Matching specific speaking styles or accents  
            - Creating voices from short audio samples
            - Fine control over speech characteristics
            
            **Kokoro TTS** - Best for:
            - High-quality pre-trained voices
            - Consistent voice quality
            - Multiple language support
            - Faster generation (no reference audio needed)
            
            **Fish Speech** - Best for:
            - Text-to-speech synthesis from text
            - Natural-sounding voice generation  
            - Customization of speech characteristics
            - Advanced audio processing controls
            
            ### 💡 Pro Tips
            - **Reference Audio**: Use clear, 3-10 second samples for ChatterboxTTS and Fish Speech
            - **Text Length**: Kokoro has a 5000 character limit, ChatterboxTTS and Fish Speech can handle longer texts
            - **Effects**: Apply reverb for space, echo for depth, pitch shift for character voices
            - **Voice Mixing**: Blend Kokoro voices with formulas like "af_heart * 0.7 + af_bella * 0.3"
            - **Presets**: Save your favorite settings for quick reuse
            - **Fish Speech Quality**: Uses clean, unprocessed output for best natural sound. Use Audio Effects section for any enhancements.
            
            ### 🎵 Audio Effects Guide
            - **Gain/Volume**: Boost or reduce overall audio level (-20 to +20 dB)
            - **3-Band EQ**: Fine-tune frequency response
              - **Bass**: Low frequencies (80-250 Hz) - warmth and fullness
              - **Mid**: Mid frequencies (250-4000 Hz) - clarity and presence  
              - **Treble**: High frequencies (4000+ Hz) - brightness and air
            - **Reverb**: Simulates room acoustics (church, hall, studio)
            - **Echo**: Adds depth and space to voice
            - **Pitch Shift**: Change voice character (±12 semitones)
            
            ### 🎛️ EQ Tips
            - **Boost Bass** (+3 to +6 dB): Warmer, fuller voice
            - **Cut Bass** (-3 to -6 dB): Cleaner, less muddy sound
            - **Boost Mid** (+2 to +4 dB): More vocal presence and clarity
            - **Boost Treble** (+2 to +5 dB): Brighter, more articulate speech
            - **Subtle is better**: Small adjustments (1-3 dB) often work best
            """)
        
        # Event handlers - No longer need to toggle visibility since all controls stay visible
        
        generate_btn.click(
            fn=generate_unified_tts,
            inputs=[
                text, tts_engine,
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