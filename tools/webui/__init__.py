from typing import Callable

import gradio as gr

from fish_speech.i18n import i18n
from tools.webui.variables import HEADER_MD, TEXTBOX_PLACEHOLDER


def build_app(inference_fct: Callable, theme: str = "dark") -> gr.Blocks:
    # Custom CSS for professional styling
    custom_css = """
    /* Global styling */
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        max-width: 1400px !important;
        margin: 0 auto !important;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .header-container h1 {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-align: center !important;
    }
    
    .header-container p {
        color: rgba(255,255,255,0.9) !important;
        text-align: center !important;
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Card styling */
    .input-card, .output-card {
        background: var(--background-fill-primary);
        border: 1px solid var(--border-color-primary);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Tab styling */
    .tab-nav button {
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        margin-right: 0.5rem !important;
    }
    
    /* Input field styling */
    .input-field textarea, .input-field input {
        border-radius: 8px !important;
        border: 2px solid var(--border-color-primary) !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .input-field textarea:focus, .input-field input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Button styling */
    .generate-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 1rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: white !important;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
        min-height: 3rem !important;
    }
    
    .generate-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Slider styling */
    .slider-container {
        margin: 1rem 0 !important;
    }
    
    /* Audio player styling */
    .audio-player {
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    
    /* Status message styling */
    .status-message {
        padding: 1rem !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        text-align: center !important;
    }
    
    .error-message {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        color: #dc2626 !important;
    }
    
    .success-message {
        background: rgba(34, 197, 94, 0.1) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        color: #059669 !important;
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
        animation: loading 1.4s infinite ease-in-out both;
    }
    
    @keyframes loading {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    /* Audio Effects Styling */
    .effects-accordion {
        background: linear-gradient(135deg, rgba(147, 51, 234, 0.05) 0%, rgba(59, 130, 246, 0.05) 100%) !important;
        border: 1px solid rgba(147, 51, 234, 0.1) !important;
        border-radius: 8px !important;
        margin: 0.5rem 0 !important;
    }
    
    .effects-section {
        background: rgba(255, 255, 255, 0.5) !important;
        border-radius: 6px !important;
        padding: 0.75rem !important;
        margin: 0.25rem 0 !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
    }
    
    .effects-checkbox {
        font-weight: 600 !important;
        color: var(--body-text-color) !important;
    }
    
    .effects-slider .gradio-slider {
        accent-color: #8b5cf6 !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .gradio-container {
            padding: 1rem !important;
        }
        
        .header-container {
            padding: 1.5rem !important;
        }
        
        .header-container h1 {
            font-size: 2rem !important;
        }
        
        .input-card, .output-card {
            padding: 1rem !important;
        }
        
        .effects-section {
            padding: 0.5rem !important;
        }
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
            spacing_size="md",
            radius_size="lg"
        ).set(
            body_background_fill="*neutral_50",
            background_fill_primary="white",
            background_fill_secondary="*neutral_100",
            border_color_primary="*neutral_200",
            shadow_drop="0 4px 6px -1px rgba(0, 0, 0, 0.1)",
        ),
        css=custom_css,
        title="🐟 Fish Speech - SUP3R Edition"
    ) as app:
        
        # Enhanced Header
        with gr.Row(elem_classes="header-container"):
            gr.HTML("""
                <div style="text-align: center;">
                    <h1>🐟 Fish Speech - SUP3R EDITION 🐟</h1>
                    <p>Advanced Text-to-Speech AI Model</p>
                    <p style="font-size: 0.95rem; opacity: 0.8;">
                        Powered by VQ-GAN and Llama Architecture | Developed by Fish Audio
                    </p>
                </div>
            """)

        # Use theme parameter for initial theme setting
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % theme,
        )

        # Main Interface
        with gr.Row():
            # Input Section
            with gr.Column(scale=5, elem_classes="input-card"):
                gr.HTML("<h3 style='margin-top: 0; color: var(--body-text-color); font-weight: 600;'>📝 Text Input</h3>")
                
                text = gr.Textbox(
                    label="Enter your text to synthesize",
                    placeholder="Type or paste your text here... The AI will convert it to natural-sounding speech.",
                    lines=8,
                    elem_classes="input-field",
                    show_label=True,
                    container=True
                )

                # Advanced Configuration
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        chunk_length = gr.Slider(
                            label="Chunk Length",
                            info="Controls iterative processing (0 = disabled)",
                            minimum=100,
                            maximum=400,
                            value=300,
                            step=8,
                            elem_classes="slider-container"
                        )

                        max_new_tokens = gr.Slider(
                            label="Max Tokens",
                            info="Maximum tokens per batch (0 = unlimited)",
                            minimum=0,
                            maximum=2048,
                            value=0,
                            step=8,
                            elem_classes="slider-container"
                        )

                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top-P",
                            info="Controls randomness in token selection",
                            minimum=0.7,
                            maximum=0.95,
                            value=0.8,
                            step=0.01,
                            elem_classes="slider-container"
                        )

                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty",
                            info="Reduces repetitive speech patterns",
                            minimum=1,
                            maximum=1.2,
                            value=1.1,
                            step=0.01,
                            elem_classes="slider-container"
                        )

                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature",
                            info="Controls speech variation and creativity",
                            minimum=0.7,
                            maximum=1.0,
                            value=0.8,
                            step=0.01,
                            elem_classes="slider-container"
                        )
                        seed = gr.Number(
                            label="Seed",
                            info="For reproducible results (0 = random)",
                            value=0,
                            precision=0
                        )

                # Reference Audio Section
                with gr.Accordion("🎤 Voice Reference", open=True):
                    gr.HTML("""
                        <div style="background: rgba(59, 130, 246, 0.05); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                            <p style="margin: 0; color: var(--body-text-color); font-size: 0.95rem;">
                                <strong>💡 Pro Tip:</strong> Upload 5-10 seconds of clear reference audio to clone a specific voice and speaking style.
                            </p>
                        </div>
                    """)
                    
                    with gr.Row():
                        reference_id = gr.Textbox(
                            label="Reference ID",
                            placeholder="Optional: Enter a reference ID for saved voices",
                            elem_classes="input-field"
                        )

                    with gr.Row():
                        use_memory_cache = gr.Radio(
                            label="Memory Cache",
                            info="Speeds up repeated generations",
                            choices=[("Enabled", "on"), ("Disabled", "off")],
                            value="on"
                        )

                    with gr.Row():
                        reference_audio = gr.Audio(
                            label="Upload Reference Audio",
                            type="filepath",
                            elem_classes="audio-player"
                        )
                    
                    with gr.Row():
                        reference_text = gr.Textbox(
                            label="Reference Text (Optional)",
                            info="Transcript of the reference audio for better results",
                            lines=2,
                            placeholder="Enter the text spoken in the reference audio...",
                            elem_classes="input-field"
                        )

                # Audio Effects Section
                with gr.Accordion("🎛️ Audio Effects", open=False, elem_classes="effects-accordion"):
                    gr.HTML("""
                        <div style="background: rgba(147, 51, 234, 0.05); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                            <p style="margin: 0; color: var(--body-text-color); font-size: 0.95rem;">
                                <strong>🎵 Pro Audio:</strong> Enhance your generated speech with professional audio effects like reverb, echo, EQ, and more.
                            </p>
                        </div>
                    """)
                    

                    
                    # Volume and Dynamics
                    with gr.Accordion("📢 Volume & Dynamics", open=False, elem_classes="effects-section"):
                        with gr.Row():
                            volume_enabled = gr.Checkbox(
                                label="Volume Adjustment",
                                value=False
                            )
                            volume_gain = gr.Slider(
                                label="Volume Gain (dB)",
                                info="Adjust overall volume",
                                minimum=-20,
                                maximum=20,
                                value=0,
                                step=1,
                                elem_classes="slider-container"
                            )
                        
                        with gr.Row():
                            compression_enabled = gr.Checkbox(
                                label="Dynamic Compression",
                                value=False
                            )
                            compression_threshold = gr.Slider(
                                label="Compression Threshold (dB)",
                                info="Level above which compression applies",
                                minimum=-40,
                                maximum=0,
                                value=-20,
                                step=1,
                                elem_classes="slider-container"
                            )
                        
                        with gr.Row():
                            compression_ratio = gr.Slider(
                                label="Compression Ratio",
                                info="Amount of compression (higher = more compressed)",
                                minimum=1,
                                maximum=10,
                                value=4,
                                step=0.5,
                                elem_classes="slider-container"
                            )
                            noise_gate_enabled = gr.Checkbox(
                                label="Noise Gate",
                                value=False
                            )
                        
                        with gr.Row():
                            noise_gate_threshold = gr.Slider(
                                label="Noise Gate Threshold (dB)",
                                info="Reduce background noise below this level",
                                minimum=-60,
                                maximum=-20,
                                value=-40,
                                step=1,
                                elem_classes="slider-container"
                            )
                    
                    # EQ Section
                    with gr.Accordion("🎚️ Equalizer", open=False, elem_classes="effects-section"):
                        with gr.Row():
                            eq_enabled = gr.Checkbox(
                                label="3-Band EQ",
                                value=False
                            )
                        
                        with gr.Row():
                            eq_bass = gr.Slider(
                                label="Bass (200Hz)",
                                info="Low frequency adjustment",
                                minimum=-12,
                                maximum=12,
                                value=0,
                                step=0.5,
                                elem_classes="slider-container"
                            )
                            eq_mid = gr.Slider(
                                label="Mid (200Hz-4kHz)",
                                info="Mid frequency adjustment",
                                minimum=-12,
                                maximum=12,
                                value=0,
                                step=0.5,
                                elem_classes="slider-container"
                            )
                            eq_treble = gr.Slider(
                                label="Treble (4kHz+)",
                                info="High frequency adjustment",
                                minimum=-12,
                                maximum=12,
                                value=0,
                                step=0.5,
                                elem_classes="slider-container"
                            )
                    
                    # Pitch and Speed
                    with gr.Accordion("🎵 Pitch & Speed", open=False, elem_classes="effects-section"):
                        with gr.Row():
                            pitch_enabled = gr.Checkbox(
                                label="Pitch Shifting",
                                value=False
                            )
                            pitch_shift = gr.Slider(
                                label="Pitch Shift (semitones)",
                                info="Shift pitch up/down (-12 to +12 semitones)",
                                minimum=-12,
                                maximum=12,
                                value=0,
                                step=0.5,
                                elem_classes="slider-container"
                            )
                        
                        with gr.Row():
                            speed_enabled = gr.Checkbox(
                                label="Speed Control",
                                value=False
                            )
                            speed_factor = gr.Slider(
                                label="Speed Factor",
                                info="Change playback speed (0.5x to 2.0x)",
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.05,
                                elem_classes="slider-container"
                            )
                    
                    # Spatial Effects
                    with gr.Accordion("🌌 Spatial Effects", open=False, elem_classes="effects-section"):
                        with gr.Row():
                            reverb_enabled = gr.Checkbox(
                                label="Reverb",
                                value=False
                            )
                            reverb_room_size = gr.Slider(
                                label="Room Size",
                                info="Simulated room size",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                elem_classes="slider-container"
                            )
                        
                        with gr.Row():
                            reverb_damping = gr.Slider(
                                label="Damping",
                                info="High frequency absorption",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                elem_classes="slider-container"
                            )
                            reverb_wet_level = gr.Slider(
                                label="Wet Level",
                                info="Amount of reverb effect",
                                minimum=0.0,
                                maximum=0.8,
                                value=0.3,
                                step=0.05,
                                elem_classes="slider-container"
                            )
                        
                        with gr.Row():
                            echo_enabled = gr.Checkbox(
                                label="Echo",
                                value=False
                            )
                            echo_delay = gr.Slider(
                                label="Echo Delay (seconds)",
                                info="Time between echoes",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.3,
                                step=0.05,
                                elem_classes="slider-container"
                            )
                        
                        with gr.Row():
                            echo_decay = gr.Slider(
                                label="Echo Decay",
                                info="Echo volume reduction",
                                minimum=0.1,
                                maximum=0.8,
                                value=0.5,
                                step=0.05,
                                elem_classes="slider-container"
                            )

            # Output Section
            with gr.Column(scale=4, elem_classes="output-card"):
                gr.HTML("<h3 style='margin-top: 0; color: var(--body-text-color); font-weight: 600;'>🔊 Generated Audio</h3>")
                
                # Status Display
                error = gr.HTML(
                    value="<div class='status-message'>Ready to generate speech. Enter your text and click Generate!</div>",
                    elem_classes="status-message"
                )
                
                # Audio Output
                audio = gr.Audio(
                    label="Your Generated Speech",
                    type="numpy",
                    interactive=False,
                    elem_classes="audio-player",
                    show_download_button=True,
                    show_share_button=True
                )

                # Generation Button
                with gr.Row():
                    generate = gr.Button(
                        value="🎧 Generate Speech",
                        variant="primary",
                        elem_classes="generate-btn",
                        size="lg"
                    )

        # Information Footer
        with gr.Row():
            gr.HTML("""
                <div style="background: var(--background-fill-secondary); padding: 1.5rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
                    <p style="margin: 0; color: var(--body-text-color-subdued); font-size: 0.9rem;">
                        <strong>Fish Speech</strong> is an open-source TTS model. 
                        <a href="https://github.com/fishaudio/fish-speech" target="_blank" style="color: #667eea;">View Source Code</a> | 
                        <a href="https://huggingface.co/fishaudio/fish-speech-1.5" target="_blank" style="color: #667eea;">Download Models</a>
                    </p>
                    <p style="margin: 0.5rem 0 0 0; color: var(--body-text-color-subdued); font-size: 0.85rem;">
                        Licensed under CC BY-NC-SA 4.0 | Please use responsibly and check your local regulations
                    </p>
                </div>
            """)


        
        generate.click(
            inference_fct,
            inputs=[
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
            ],
            outputs=[audio, error],
            concurrency_limit=1,
        )

    return app
