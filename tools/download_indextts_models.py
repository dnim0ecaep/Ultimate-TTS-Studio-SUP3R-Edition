#!/usr/bin/env python3
"""
IndexTTS Model Downloader
Downloads required model files from HuggingFace Hub for IndexTTS integration.
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    import requests
except ImportError:
    print("❌ Missing dependencies. Please install:")
    print("pip install huggingface_hub requests")
    sys.exit(1)

def download_indextts_models():
    """Download IndexTTS models from HuggingFace"""
    
    repo_id = "IndexTeam/IndexTTS-1.5"
    model_dir = Path("indextts/checkpoints")
    
    # Create directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # List of required files
    required_files = [
        "config.yaml",
        "gpt.pth",
        "bigvgan_generator.pth", 
        "bpe.model"
    ]
    
    print("🎯 IndexTTS Model Downloader")
    print("=" * 50)
    print(f"📁 Download directory: {model_dir.absolute()}")
    print(f"🏢 Repository: {repo_id}")
    print()
    
    for filename in required_files:
        file_path = model_dir / filename
        
        # Skip if file already exists
        if file_path.exists():
            print(f"✅ {filename} - Already exists")
            continue
            
        try:
            print(f"⬇️  Downloading {filename}...")
            
            # Download file
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False
            )
            
            print(f"✅ {filename} - Downloaded successfully")
            
        except Exception as e:
            print(f"❌ {filename} - Download failed: {e}")
            return False
    
    print()
    print("🎉 IndexTTS models downloaded successfully!")
    print()
    print("📝 Next steps:")
    print("1. Start the TTS Studio: python launch.py")
    print("2. Click 'Load IndexTTS' in the Model Management section")
    print("3. Upload a reference audio file")
    print("4. Enter your text and generate speech!")
    print()
    
    return True

if __name__ == "__main__":
    if download_indextts_models():
        sys.exit(0)
    else:
        print("\n❌ Download failed. Please check your internet connection and try again.")
        sys.exit(1) 