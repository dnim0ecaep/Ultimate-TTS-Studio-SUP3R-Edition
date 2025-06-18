# IndexTTS Package
"""
IndexTTS - Industrial-level controllable TTS system
"""

# Make infer module available at package root level
from indextts.indextts.infer import IndexTTS
from indextts.indextts import infer

# Create a compatibility layer for the old import style
import sys
sys.modules['indextts.infer'] = infer

__version__ = "1.5.0"
__all__ = ["IndexTTS", "infer"] 