import os
import threading
from argparse import ArgumentParser
from pathlib import Path

import pyrootutils
import torch
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from tools.webui import build_app
from tools.webui.inference import get_inference_wrapper
from tools.webui.audio_effects import create_audio_effects_processor

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/openaudio-s1-mini",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/openaudio-s1-mini/codec.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="dark")
    parser.add_argument("--skip-dry-run", action="store_true", default=True, help="Skip the warm-up dry run")
    parser.add_argument("--skip-model-loading", action="store_true", help="Skip model loading entirely")

    return parser.parse_args()


def load_models_in_background(args):
    """Load models in a background thread"""
    global inference_engine, app_inference_fct
    
    try:
        logger.info("Loading Llama model...")
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=args.llama_checkpoint_path,
            device=args.device,
            precision=args.precision,
            compile=args.compile,
        )

        logger.info("Loading VQ-GAN model...")
        decoder_model = load_decoder_model(
            config_name=args.decoder_config_name,
            checkpoint_path=args.decoder_checkpoint_path,
            device=args.device,
        )

        logger.info("Decoder model loaded...")

        # Create the inference engine
        inference_engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            compile=args.compile,
            precision=args.precision,
        )

        if not args.skip_dry_run:
            logger.info("Warming up...")
            # Dry run to check if the model is loaded correctly and avoid the first-time latency
            list(
                inference_engine.inference(
                    ServeTTSRequest(
                        text="Hello world.",
                        references=[],
                        reference_id=None,
                        max_new_tokens=1024,
                        chunk_length=200,
                        top_p=0.7,
                        repetition_penalty=1.5,
                        temperature=0.7,
                        format="wav",
                    )
                )
            )
            logger.info("Warming up done!")
        
        # Update the global inference function
        app_inference_fct = get_inference_wrapper(inference_engine)
        logger.info("🎉 Models loaded successfully! The web UI is now fully functional.")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        inference_engine = None
        app_inference_fct = None


if __name__ == "__main__":
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    # Check if MPS or CUDA is available
    if torch.backends.mps.is_available():
        args.device = "mps"
        logger.info("mps is available, running on mps.")
    elif not torch.cuda.is_available():
        logger.info("CUDA is not available, running on CPU.")
        args.device = "cpu"

    # Global variables to hold the inference engine and function
    inference_engine = None
    app_inference_fct = None
    audio_effects_processor = create_audio_effects_processor()

    def dynamic_inference_wrapper(
        text,
        reference_id,
        reference_audio,
        reference_text,
        max_new_tokens,
        chunk_length,
        top_p,
        repetition_penalty,
        temperature,
        seed,
        use_memory_cache,
        # Audio Effects Parameters
        reverb_enabled,
        reverb_room_size,
        reverb_damping,
        reverb_wet_level,
        echo_enabled,
        echo_delay,
        echo_decay,
        eq_enabled,
        eq_bass,
        eq_mid,
        eq_treble,
        pitch_enabled,
        pitch_shift,
        speed_enabled,
        speed_factor,
        volume_enabled,
        volume_gain,
        compression_enabled,
        compression_threshold,
        compression_ratio,
        noise_gate_enabled,
        noise_gate_threshold,
    ):
        """Wrapper that checks if models are loaded before calling inference"""
        if app_inference_fct is None:
            loading_message = """
            <div class='status-message' style='background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); color: #2563eb;'>
                🔄 <strong>AI Models Loading...</strong><br/>
                <small>This may take a few minutes on first launch. The interface will be fully functional once loading completes.</small>
                <div style='margin-top: 0.5rem;'>
                    <div class='loading-dots' style='margin-right: 0.2rem;'>●</div>
                    <div class='loading-dots' style='animation-delay: 0.2s; margin-right: 0.2rem;'>●</div>
                    <div class='loading-dots' style='animation-delay: 0.4s;'>●</div>
                </div>
            </div>
            """
            return None, loading_message
        
        if not text or not text.strip():
            empty_message = """
            <div class='status-message' style='background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3); color: #d97706;'>
                ⚠️ <strong>Please enter some text</strong><br/>
                <small>Type or paste the text you want to convert to speech in the input field above.</small>
            </div>
            """
            return None, empty_message
            
        try:
            # Show processing message
            processing_message = """
            <div class='status-message' style='background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); color: #2563eb;'>
                🎵 <strong>Generating Speech...</strong><br/>
                <small>AI is processing your text and creating natural-sounding audio.</small>
            </div>
            """
            
            result = app_inference_fct(
                text,
                reference_id,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                seed,
                use_memory_cache,
            )
            
            # Success message
            if result[0] is not None:  # Audio generated successfully
                # Apply audio effects if any are enabled
                audio_data = result[0]
                sample_rate = audio_data[0] if isinstance(audio_data, tuple) else 22050
                audio_array = audio_data[1] if isinstance(audio_data, tuple) else audio_data
                
                # Check if any effects are enabled
                effects_enabled = any([
                    reverb_enabled, echo_enabled, eq_enabled, pitch_enabled, 
                    speed_enabled, volume_enabled, compression_enabled, noise_gate_enabled
                ])
                
                if effects_enabled:
                    try:
                        # Apply audio effects
                        processed_sample_rate, processed_audio = audio_effects_processor.apply_effects(
                            audio_array,
                            sample_rate,
                            reverb_enabled=reverb_enabled,
                            reverb_room_size=reverb_room_size,
                            reverb_damping=reverb_damping,
                            reverb_wet_level=reverb_wet_level,
                            echo_enabled=echo_enabled,
                            echo_delay=echo_delay,
                            echo_decay=echo_decay,
                            eq_enabled=eq_enabled,
                            eq_bass=eq_bass,
                            eq_mid=eq_mid,
                            eq_treble=eq_treble,
                            pitch_enabled=pitch_enabled,
                            pitch_shift=pitch_shift,
                            speed_enabled=speed_enabled,
                            speed_factor=speed_factor,
                            volume_enabled=volume_enabled,
                            volume_gain=volume_gain,
                            compression_enabled=compression_enabled,
                            compression_threshold=compression_threshold,
                            compression_ratio=compression_ratio,
                            noise_gate_enabled=noise_gate_enabled,
                            noise_gate_threshold=noise_gate_threshold,
                        )
                        
                        # Return processed audio
                        processed_result = (processed_sample_rate, processed_audio)
                        success_message = """
                        <div class='status-message success-message'>
                            ✅ <strong>Speech Generated with Effects!</strong><br/>
                            <small>Your audio has been enhanced with professional effects. You can play, download, or share it using the controls below.</small>
                        </div>
                        """
                        return processed_result, success_message
                        
                    except Exception as e:
                        logger.error(f"Error applying audio effects: {e}")
                        # Return original audio if effects fail
                        warning_message = f"""
                        <div class='status-message' style='background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3); color: #d97706;'>
                            ⚠️ <strong>Effects Processing Failed</strong><br/>
                            <small>Audio generated successfully, but effects could not be applied: {str(e)}</small><br/>
                            <small>Returning original audio without effects.</small>
                        </div>
                        """
                        return result[0], warning_message
                else:
                    success_message = """
                    <div class='status-message success-message'>
                        ✅ <strong>Speech Generated Successfully!</strong><br/>
                        <small>Your audio is ready. You can play, download, or share it using the controls below.</small>
                    </div>
                    """
                    return result[0], success_message
            else:
                return result
                
        except Exception as e:
            error_message = f"""
            <div class='status-message error-message'>
                ❌ <strong>Generation Failed</strong><br/>
                <small>Error: {str(e)}</small><br/>
                <small>Please try again or adjust your settings.</small>
            </div>
            """
            return None, error_message

    # Launch web UI first
    logger.info("🚀 Launching web UI...")
    app = build_app(dynamic_inference_wrapper, args.theme)
    
    # Start model loading in background if not skipped
    if not args.skip_model_loading:
        logger.info("📦 Starting model loading in background...")
        model_thread = threading.Thread(target=load_models_in_background, args=(args,))
        model_thread.daemon = True
        model_thread.start()
    else:
        logger.info("Skipping model loading as requested.")

    # Launch the app
    app.launch(show_api=True, inbrowser=True)
