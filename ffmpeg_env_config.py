"""
FFmpeg Configuration for Virtual Environment
Uses FFmpeg binaries installed in the virtual environment via imageio-ffmpeg.
"""

import os
import sys

def configure_env_ffmpeg():
    """Configure pydub to use FFmpeg from the virtual environment."""
    try:
        # Get FFmpeg path from imageio-ffmpeg (installed in env)
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            # Configure pydub to use this FFmpeg
            from pydub import AudioSegment
            AudioSegment.converter = ffmpeg_path
            AudioSegment.ffmpeg = ffmpeg_path
            AudioSegment.ffprobe = ffmpeg_path.replace('ffmpeg', 'ffprobe')
            
            print(f"✅ Using env FFmpeg: {ffmpeg_path}")
            return True
        else:
            print("⚠️ imageio-ffmpeg not found - install it with: pip install imageio-ffmpeg")
            return False
            
    except ImportError:
        print("⚠️ imageio-ffmpeg not installed - MP3 conversion will use system FFmpeg if available")
        return False
    except Exception as e:
        print(f"⚠️ Error configuring env FFmpeg: {e}")
        return False

# Auto-configure when imported
if __name__ != "__main__":
    configure_env_ffmpeg() 