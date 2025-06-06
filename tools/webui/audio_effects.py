import numpy as np
import librosa
from typing import Tuple, Optional
from scipy import signal
import warnings

warnings.filterwarnings("ignore")


class AudioEffects:
    """Audio effects processor for post-processing generated speech"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def apply_effects(
        self,
        audio: np.ndarray,
        sample_rate: int,
        # Reverb effects
        reverb_enabled: bool = False,
        reverb_room_size: float = 0.5,
        reverb_damping: float = 0.5,
        reverb_wet_level: float = 0.3,
        # Echo effects  
        echo_enabled: bool = False,
        echo_delay: float = 0.3,
        echo_decay: float = 0.5,
        # EQ effects
        eq_enabled: bool = False,
        eq_bass: float = 0.0,
        eq_mid: float = 0.0,
        eq_treble: float = 0.0,
        # Pitch effects
        pitch_enabled: bool = False,
        pitch_shift: float = 0.0,
        # Speed effects
        speed_enabled: bool = False,
        speed_factor: float = 1.0,
        # Volume effects
        volume_enabled: bool = False,
        volume_gain: float = 0.0,
        # Compression
        compression_enabled: bool = False,
        compression_threshold: float = -20.0,
        compression_ratio: float = 4.0,
        # Noise gate
        noise_gate_enabled: bool = False,
        noise_gate_threshold: float = -40.0,
    ) -> Tuple[int, np.ndarray]:
        """Apply selected audio effects to the audio signal"""
        
        processed_audio = audio.copy()
        
        # Ensure audio is float32 and in the right range
        if processed_audio.dtype != np.float32:
            processed_audio = processed_audio.astype(np.float32)
        
        # Normalize to [-1, 1] if needed
        if np.max(np.abs(processed_audio)) > 1.0:
            processed_audio = processed_audio / np.max(np.abs(processed_audio))
        
        try:
            # Apply effects in order
            if noise_gate_enabled:
                processed_audio = self._apply_noise_gate(processed_audio, noise_gate_threshold)
            
            if compression_enabled:
                processed_audio = self._apply_compression(processed_audio, compression_threshold, compression_ratio)
            
            if eq_enabled:
                processed_audio = self._apply_eq(processed_audio, sample_rate, eq_bass, eq_mid, eq_treble)
            
            if pitch_enabled and abs(pitch_shift) > 0.1:
                processed_audio = self._apply_pitch_shift(processed_audio, sample_rate, pitch_shift)
            
            if speed_enabled and abs(speed_factor - 1.0) > 0.05:
                processed_audio = self._apply_speed_change(processed_audio, sample_rate, speed_factor)
            
            if echo_enabled:
                processed_audio = self._apply_echo(processed_audio, sample_rate, echo_delay, echo_decay)
            
            if reverb_enabled:
                processed_audio = self._apply_reverb(processed_audio, sample_rate, reverb_room_size, reverb_damping, reverb_wet_level)
            
            if volume_enabled and abs(volume_gain) > 0.1:
                processed_audio = self._apply_volume(processed_audio, volume_gain)
            
            # Final safety check
            processed_audio = np.clip(processed_audio, -1.0, 1.0)
            
        except Exception as e:
            print(f"Error applying effects: {e}")
            return sample_rate, audio  # Return original audio on error
        
        return sample_rate, processed_audio
    
    def _apply_reverb(self, audio: np.ndarray, sample_rate: int, room_size: float, damping: float, wet_level: float) -> np.ndarray:
        """Apply simple reverb effect using multiple delays"""
        # Create multiple delay lines for reverb
        delays = [0.03, 0.05, 0.07, 0.11, 0.13, 0.17]  # Prime numbers for natural sound
        decays = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        
        reverb_signal = np.zeros_like(audio)
        
        for delay, decay in zip(delays, decays):
            delay_samples = int(delay * sample_rate * room_size)
            if delay_samples > 0 and delay_samples < len(audio):
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples] * decay * (1 - damping)
                reverb_signal += delayed
        
        # Mix wet and dry signals
        return audio * (1 - wet_level) + reverb_signal * wet_level
    
    def _apply_echo(self, audio: np.ndarray, sample_rate: int, delay: float, decay: float) -> np.ndarray:
        """Apply echo effect"""
        delay_samples = int(delay * sample_rate)
        if delay_samples <= 0 or delay_samples >= len(audio):
            return audio
        
        echo_signal = np.zeros_like(audio)
        echo_signal[delay_samples:] = audio[:-delay_samples] * decay
        
        return audio + echo_signal
    
    def _apply_eq(self, audio: np.ndarray, sample_rate: int, bass: float, mid: float, treble: float) -> np.ndarray:
        """Apply 3-band equalizer"""
        if abs(bass) < 0.1 and abs(mid) < 0.1 and abs(treble) < 0.1:
            return audio
        
        # Define frequency bands
        bass_freq = 200
        treble_freq = 4000
        
        # Design filters
        nyquist = sample_rate / 2
        
        try:
            # Bass filter (low shelf)
            if abs(bass) > 0.1:
                bass_gain = 10 ** (bass / 20)  # Convert dB to linear
                b_bass, a_bass = signal.iirfilter(2, bass_freq / nyquist, btype='lowpass', ftype='butter')
                bass_filtered = signal.filtfilt(b_bass, a_bass, audio)
                audio = audio + (bass_filtered - audio) * (bass_gain - 1)
            
            # Treble filter (high shelf)
            if abs(treble) > 0.1:
                treble_gain = 10 ** (treble / 20)
                b_treble, a_treble = signal.iirfilter(2, treble_freq / nyquist, btype='highpass', ftype='butter')
                treble_filtered = signal.filtfilt(b_treble, a_treble, audio)
                audio = audio + (treble_filtered - audio) * (treble_gain - 1)
            
            # Mid filter (bandpass)
            if abs(mid) > 0.1:
                mid_gain = 10 ** (mid / 20)
                b_mid, a_mid = signal.iirfilter(2, [bass_freq / nyquist, treble_freq / nyquist], btype='bandpass', ftype='butter')
                mid_filtered = signal.filtfilt(b_mid, a_mid, audio)
                audio = audio + mid_filtered * (mid_gain - 1)
                
        except Exception as e:
            print(f"EQ error: {e}")
            return audio
        
        return audio
    
    def _apply_pitch_shift(self, audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
        """Apply pitch shifting using librosa"""
        try:
            return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)
        except Exception as e:
            print(f"Pitch shift error: {e}")
            return audio
    
    def _apply_speed_change(self, audio: np.ndarray, sample_rate: int, speed_factor: float) -> np.ndarray:
        """Apply speed change using librosa"""
        try:
            return librosa.effects.time_stretch(audio, rate=speed_factor)
        except Exception as e:
            print(f"Speed change error: {e}")
            return audio
    
    def _apply_volume(self, audio: np.ndarray, gain_db: float) -> np.ndarray:
        """Apply volume gain in dB"""
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    
    def _apply_compression(self, audio: np.ndarray, threshold_db: float, ratio: float) -> np.ndarray:
        """Apply dynamic range compression"""
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Simple compression algorithm
        compressed = audio.copy()
        mask = np.abs(audio) > threshold_linear
        
        # Apply compression to signals above threshold
        excess = np.abs(audio[mask]) - threshold_linear
        compressed_excess = excess / ratio
        compressed[mask] = np.sign(audio[mask]) * (threshold_linear + compressed_excess)
        
        return compressed
    
    def _apply_noise_gate(self, audio: np.ndarray, threshold_db: float) -> np.ndarray:
        """Apply noise gate to reduce low-level noise"""
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Simple gate - attenuate signals below threshold
        gated = audio.copy()
        mask = np.abs(audio) < threshold_linear
        gated[mask] *= 0.1  # Reduce by 20dB instead of complete cut
        
        return gated


def create_audio_effects_processor() -> AudioEffects:
    """Factory function to create an audio effects processor"""
    return AudioEffects() 