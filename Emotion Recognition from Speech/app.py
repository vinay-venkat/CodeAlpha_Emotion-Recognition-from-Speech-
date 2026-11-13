"""Interactive Gradio web app for Speech Emotion Recognition with Glassmorphism UI."""

from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import PathConfig
from src.features import extract_mfcc_sequence, extract_pooled_features, load_and_preprocess_audio
from src.predict import find_default_checkpoint, load_predictor


CHECKPOINT_PATH = find_default_checkpoint()
_CACHED_BUNDLE = None

EMOTION_META = {
    "angry": {"emoji": "😡", "color": "#f87171", "bg": "rgba(248, 113, 113, 0.15)", "desc": "High intensity, elevated pitch & rapid spectral flux"},
    "happy": {"emoji": "😄", "color": "#fbbf24", "bg": "rgba(251, 191, 36, 0.15)", "desc": "Bright timbre, dynamic pitch contour & energetic harmonics"},
    "sad": {"emoji": "😢", "color": "#60a5fa", "bg": "rgba(96, 165, 250, 0.15)", "desc": "Subdued energy, downward pitch contour & lower RMS"},
    "neutral": {"emoji": "😐", "color": "#94a3b8", "bg": "rgba(148, 163, 184, 0.15)", "desc": "Even tempo, stable baseline pitch & balanced spectrum"},
    "fearful": {"emoji": "😨", "color": "#c084fc", "bg": "rgba(192, 132, 252, 0.15)", "desc": "High fundamental frequency, vocal tremor & jitter"},
    "disgust": {"emoji": "🤢", "color": "#34d399", "bg": "rgba(52, 211, 153, 0.15)", "desc": "Low-to-mid frequency resonance & harsh spectral tilt"},
    "surprised": {"emoji": "😲", "color": "#38bdf8", "bg": "rgba(56, 189, 248, 0.15)", "desc": "Sharp pitch inflection & rapid energy surge"},
}

GLASS_CSS = """
/* Glassmorphic Global Theme */
body, .gradio-container {
    background: radial-gradient(circle at 10% 20%, rgba(79, 70, 229, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(219, 39, 119, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(14, 165, 233, 0.08) 0%, transparent 60%),
                #090d16 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif !important;
    color: #e2e8f0 !important;
}

/* Glass Card styling for Gradio blocks */
.glass-panel, .gr-panel, .gr-box, div[data-testid="block"] {
    background: rgba(17, 24, 39, 0.6) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 18px !important;
    box-shadow: 0 12px 35px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 0 rgba(255, 255, 255, 0.12) !important;
}

/* Glowing Primary Button */
.glow-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 28px !important;
    box-shadow: 0 0 25px rgba(124, 58, 237, 0.45), 0 8px 20px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.glow-btn:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 0 35px rgba(236, 72, 153, 0.6), 0 12px 25px rgba(0, 0, 0, 0.4) !important;
}

/* Custom Header Badge */
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(99, 102, 241, 0.18);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #a5b4fc;
    margin-bottom: 12px;
}

/* Prediction Result Glass Banner */
.hero-result-card {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 15px 40px -10px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    margin-top: 10px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

/* Audio Player and inputs styling */
audio {
    border-radius: 12px !important;
    filter: invert(0.88) hue-rotate(180deg) !important;
}

/* Footer styling */
.glass-footer {
    text-align: center;
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 25px;
    padding-top: 15px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}
"""


def get_bundle():
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is None:
        ckpt = find_default_checkpoint()
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Trained model checkpoint not found in {PathConfig().checkpoint_dir}. "
                "Please run `python -m src.train` first."
            )
        _CACHED_BUNDLE = load_predictor(ckpt)
    return _CACHED_BUNDLE


def make_diagnostic_plot(signal: np.ndarray, sample_rate: int, n_mfcc: int = 40):
    """Plot preprocessed speech waveform and MFCC spectrogram in dark glass aesthetic."""
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.0), facecolor="#090d16")

    # Waveform plot
    times = np.linspace(0, len(signal) / sample_rate, num=len(signal))
    axes[0].set_facecolor("#0f172a")
    axes[0].plot(times, signal, color="#38bdf8", linewidth=0.9, alpha=0.95)
    axes[0].fill_between(times, signal, color="#38bdf8", alpha=0.15)
    axes[0].set_title("Preprocessed Speech Waveform (Normalized)", color="#f1f5f9", fontsize=11, fontweight="bold", pad=8)
    axes[0].set_xlabel("Time (seconds)", color="#94a3b8", fontsize=9)
    axes[0].set_ylabel("Amplitude", color="#94a3b8", fontsize=9)
    axes[0].tick_params(colors="#64748b", labelsize=8)
    border_color = (1.0, 1.0, 1.0, 0.08)
    for spine in axes[0].spines.values():
        spine.set_color(border_color)

    # MFCC Spectrogram
    axes[1].set_facecolor("#0f172a")
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc)
    img = librosa.display.specshow(
        mfcc,
        sr=sample_rate,
        x_axis="time",
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set_title(f"MFCC Feature Map ({n_mfcc} Cepstral Bands)", color="#f1f5f9", fontsize=11, fontweight="bold", pad=8)
    axes[1].set_ylabel("MFCC Index", color="#94a3b8", fontsize=9)
    axes[1].set_xlabel("Time (seconds)", color="#94a3b8", fontsize=9)
    axes[1].tick_params(colors="#64748b", labelsize=8)
    for spine in axes[1].spines.values():
        spine.set_color(border_color)

    cbar = fig.colorbar(img, ax=axes[1], format="%+2.0f", pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#64748b", labelsize=8)
    cbar.outline.set_color(border_color)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")

    plt.tight_layout()
    return fig


def build_hero_html(top_emotion: str, confidence: float, runner_up: Tuple[str, float]) -> str:
    meta = EMOTION_META.get(top_emotion, {"emoji": "✨", "color": "#818cf8", "bg": "rgba(129, 140, 248, 0.15)", "desc": "Classified acoustic pattern"})
    runner_meta = EMOTION_META.get(runner_up[0], {"emoji": "🔹", "color": "#94a3b8"})
    
    return f"""
    <div class="hero-result-card" style="border-left: 5px solid {meta['color']};">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span style="font-size: 2.8rem; line-height: 1; filter: drop-shadow(0 4px 10px rgba(0,0,0,0.4));">{meta['emoji']}</span>
                <div>
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1.5px; color: #94a3b8; font-weight: 600;">Predicted State</div>
                    <div style="font-size: 2rem; font-weight: 800; color: {meta['color']}; text-shadow: 0 0 20px {meta['bg']}; line-height: 1.1;">
                        {top_emotion.upper()}
                    </div>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 2rem; font-weight: 900; color: #f8fafc; font-family: monospace;">
                    {confidence:.1f}%
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8;">Confidence Score</div>
            </div>
        </div>

        <!-- Progress bar -->
        <div style="width: 100%; height: 8px; background: rgba(255, 255, 255, 0.08); border-radius: 999px; overflow: hidden; margin-top: 4px;">
            <div style="width: {confidence:.1f}%; height: 100%; background: linear-gradient(90deg, #6366f1, {meta['color']}); border-radius: 999px; box-shadow: 0 0 12px {meta['color']};"></div>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">
            <span>ℹ️ <em>{meta['desc']}</em></span>
            <span style="color: #94a3b8;">Runner-up: <strong style="color: {runner_meta['color']};">{runner_up[0].capitalize()}</strong> ({runner_up[1]:.1f}%)</span>
        </div>
    </div>
    """


def analyze_speech(audio_filepath: str):
    if not audio_filepath:
        return {}, None, "<div style='color: #f87171; padding: 10px;'>⚠️ Please upload an audio file or record speech.</div>"

    try:
        model_obj, audio_cfg, classes, framework = get_bundle()
    except FileNotFoundError as err:
        return {}, None, f"<div style='color: #f87171;'>❌ {err}</div>"

    signal = load_and_preprocess_audio(Path(audio_filepath), audio_cfg)

    if framework == "sklearn":
        feat = extract_pooled_features(signal, audio_cfg).reshape(1, -1)
        probs = model_obj.predict_proba(feat)[0]
    else:
        import torch
        model, scaler, device = model_obj
        seq = extract_mfcc_sequence(signal, audio_cfg)
        seq_scaled = scaler.transform(seq[np.newaxis, ...])
        with torch.no_grad():
            logits = model(torch.tensor(seq_scaled, dtype=torch.float32, device=device))
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    prob_dict = {cls_name: float(p) for cls_name, p in zip(classes, probs)}
    sorted_items = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    top_emotion, top_prob = sorted_items[0]
    runner_up = (sorted_items[1][0], sorted_items[1][1] * 100) if len(sorted_items) > 1 else ("none", 0.0)

    fig = make_diagnostic_plot(signal, audio_cfg.sample_rate, n_mfcc=audio_cfg.n_mfcc)
    hero_html = build_hero_html(top_emotion, top_prob * 100, runner_up)
    return prob_dict, fig, hero_html


def create_app() -> gr.Blocks:
    sample_examples = []
    candidates = [
        "data/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav",
        "data/TESS Toronto emotional speech set data/OAF_happy/OAF_back_happy.wav",
        "data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav",
        "data/RAVDESS/Actor_01/03-01-04-01-01-01-01.wav",
    ]
    for c in candidates:
        if Path(c).exists():
            sample_examples.append([c])

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="pink",
        neutral_hue="slate",
    ).set(
        body_background_fill="#090d16",
        block_background_fill="rgba(17, 24, 39, 0.6)",
        block_border_width="1px",
        block_border_color="rgba(255, 255, 255, 0.09)",
        block_radius="18px",
        button_primary_background_fill="linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%)",
        button_primary_text_color="#ffffff",
    )

    with gr.Blocks(title="Speech Emotion Recognition AI", theme=theme, css=GLASS_CSS) as demo:
        # Injected Glassmorphic Style Tag for guaranteed stylesheet application
        gr.HTML(f"<style>{GLASS_CSS}</style>")

        # Header with Glassmorphic badge and glowing title
        gr.HTML(
            """
            <div style="text-align: center; padding: 25px 15px 15px 15px;">
                <div class="header-badge">
                    <span>⚡ AI-Powered Audio Intelligence</span>
                    <span style="opacity: 0.4;">|</span>
                    <span>272 Acoustic Dimensions</span>
                </div>
                <h1 style="font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa 0%, #c084fc 45%, #f472b6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 10px 0; letter-spacing: -0.5px;">
                    Speech Emotion Recognition (SER)
                </h1>
                <p style="font-size: 1.05rem; color: #94a3b8; max-width: 680px; margin: 0 auto; line-height: 1.5;">
                    Real-time human affective computing from vocal patterns. Detects 
                    <strong style="color: #f1f5f9;">7 emotional states</strong> using 
                    <strong>MFCCs, Spectral Flux, Prosody & Deep Neural Networks</strong>.
                </p>
            </div>
            """
        )

        with gr.Row():
            # Left column: Audio input & Action
            with gr.Column(scale=5, elem_classes="glass-panel"):
                gr.Markdown("### 🎤 Vocal Input Stream")
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record voice or drop audio file (.wav / .mp3 / .flac)",
                )
                submit_btn = gr.Button("🔮 Analyze Speech Emotion", variant="primary", elem_classes="glow-btn")

                # Dynamic Glass Hero Prediction Banner
                result_banner = gr.HTML(
                    """
                    <div style="text-align: center; padding: 24px; color: #64748b; border: 1px dashed rgba(255,255,255,0.1); border-radius: 16px; margin-top: 10px;">
                        <span>🎙️ Record audio or select an example below, then click <strong>Analyze Speech Emotion</strong></span>
                    </div>
                    """
                )

                if sample_examples:
                    gr.Markdown("#### 🎧 Quick Presets (Click to Load):")
                    gr.Examples(
                        examples=sample_examples,
                        inputs=[audio_input],
                        label="Sample Benchmark Recordings",
                    )

            # Right column: Probability distribution & Visualizations
            with gr.Column(scale=6, elem_classes="glass-panel"):
                gr.Markdown("### 📊 Affective Probability Spectrum")
                label_output = gr.Label(num_top_classes=7, label="Class Likelihood Distribution")

                gr.Markdown("### 📈 Acoustic Waveform & Cepstral Map")
                plot_output = gr.Plot(label="Diagnostic Spectral Analysis")

        submit_btn.click(
            fn=analyze_speech,
            inputs=[audio_input],
            outputs=[label_output, plot_output, result_banner],
        )

        gr.HTML(
            """
            <div class="glass-footer">
                Built with <strong>PyTorch • Librosa • Scikit-Learn • Gradio</strong> &nbsp;|&nbsp; 
                Trained on <strong>RAVDESS, TESS & EMO-DB</strong> &nbsp;|&nbsp; 
                Speaker-Independent Group Validation
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch()
