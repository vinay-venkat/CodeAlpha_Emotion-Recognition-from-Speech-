"""Generate synthetic RAVDESS-formatted audio clips for end-to-end pipeline smoke testing."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf

from src.config import PathConfig


# Distinct frequency/modulation signatures per emotion code so a model can learn real separation
EMOTION_SYNTH_PARAMS = {
    "01": (160.0, 2.0, 0.25),  # neutral: steady low pitch
    "03": (290.0, 6.0, 0.50),  # happy: brighter pitch + lively vibrato
    "04": (130.0, 1.5, 0.18),  # sad: low pitch, slow modulation, softer
    "05": (240.0, 14.0, 0.75), # angry: strong amplitude + rapid modulation
    "06": (330.0, 11.0, 0.40), # fearful: high pitch + tremor
    "07": (185.0, 4.5, 0.35),  # disgust: mid-low nasal buzz
    "08": (360.0, 8.0, 0.55),  # surprised: rising bright tone
}


def synthesize_clip(
    base_freq: float,
    mod_freq: float,
    amp: float,
    rng: np.random.Generator,
    sample_rate: int = 22050,
    duration: float = 2.5,
) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    freq_jitter = rng.uniform(-12.0, 12.0)
    carrier = np.sin(2 * np.pi * (base_freq + freq_jitter) * t)
    harmonic = 0.4 * np.sin(2 * np.pi * 2.0 * (base_freq + freq_jitter) * t)
    envelope = 0.65 + 0.35 * np.sin(2 * np.pi * mod_freq * t)
    noise = 0.03 * rng.normal(size=len(t))
    signal = amp * envelope * (carrier + harmonic) + noise
    return np.clip(signal, -1.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic RAVDESS-style demo dataset.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PathConfig().data_dir / "demo" / "ravdess",
        help="Destination folder for synthetic wav files.",
    )
    parser.add_argument("--actors", type=int, default=6, help="Number of synthetic speakers.")
    parser.add_argument("--clips-per-emotion", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    created = 0

    for actor_idx in range(1, args.actors + 1):
        actor_dir = args.out_dir / f"Actor_{actor_idx:02d}"
        actor_dir.mkdir(parents=True, exist_ok=True)
        for emo_code, (freq, mod, amp) in EMOTION_SYNTH_PARAMS.items():
            for rep in range(1, args.clips_per_emotion + 1):
                filename = f"03-01-{emo_code}-01-01-{rep:02d}-{actor_idx:02d}.wav"
                audio = synthesize_clip(freq, mod, amp, rng)
                sf.write(actor_dir / filename, audio, samplerate=22050)
                created += 1

    print(f"Created {created} synthetic WAV files in: {args.out_dir}")
    print("Run a quick smoke test training with:")
    print(f"  python -m src.train --raw-dir \"{args.out_dir.parent}\" --datasets ravdess --epochs 5 --batch-size 16")


if __name__ == "__main__":
    main()

