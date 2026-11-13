"""Predict emotion from a speech audio file using a trained model checkpoint."""

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np

from src.config import AudioConfig, PathConfig
from src.features import extract_pooled_features, load_and_preprocess_audio


def find_default_checkpoint() -> Path:
    ckpt_dir = PathConfig().checkpoint_dir
    for name in ("best_model.joblib", "best_model.pt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    return ckpt_dir / "best_model.joblib"


def load_predictor(checkpoint_path: Path) -> Tuple[Any, AudioConfig, list[str], str]:
    """Load model artifact (.joblib or .pt) and return (model, audio_cfg, classes, framework)."""
    if checkpoint_path.suffix == ".joblib" or (checkpoint_path.parent / (checkpoint_path.stem + ".joblib")).exists():
        actual_path = checkpoint_path if checkpoint_path.exists() else (checkpoint_path.parent / (checkpoint_path.stem + ".joblib"))
        bundle = joblib.load(actual_path)
        audio_cfg = AudioConfig(**bundle["audio_config"])
        classes = list(bundle["classes"])
        model = bundle["pipeline"]
        return model, audio_cfg, classes, "sklearn"

    # PyTorch checkpoint fallback
    try:
        import torch
        from src.features import SequenceFeatureScaler
        from src.models import build_model
    except (ImportError, OSError) as err:
        raise RuntimeError(f"PyTorch is required to load {checkpoint_path}: {err}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    audio_cfg = AudioConfig(**ckpt["audio_config"])
    classes = ckpt["classes"]
    scaler = SequenceFeatureScaler.from_state(ckpt["scaler_mean"], ckpt["scaler_std"])
    model = build_model(
        model_type=ckpt["model_type"],
        input_dim=audio_cfg.n_features,
        num_classes=len(classes),
        dropout=ckpt.get("dropout", 0.35),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return (model, scaler, device), audio_cfg, classes, "torch"


def predict_audio_file(
    audio_path: Path,
    checkpoint_path: Path,
) -> Tuple[str, Dict[str, float]]:
    """Return `(predicted_emotion, probabilities_by_class)` for an audio file."""
    model_obj, audio_cfg, classes, framework = load_predictor(checkpoint_path)
    signal = load_and_preprocess_audio(audio_path, audio_cfg)

    if framework == "sklearn":
        feat = extract_pooled_features(signal, audio_cfg).reshape(1, -1)
        probs = model_obj.predict_proba(feat)[0]
    else:
        import torch
        from src.features import extract_mfcc_sequence
        model, scaler, device = model_obj
        seq = extract_mfcc_sequence(signal, audio_cfg)
        seq_scaled = scaler.transform(seq[np.newaxis, ...])
        with torch.no_grad():
            logits = model(torch.tensor(seq_scaled, dtype=torch.float32, device=device))
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    prob_dict = {cls_name: float(prob) for cls_name, prob in zip(classes, probs)}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True))
    top_emotion = next(iter(sorted_probs))
    return top_emotion, sorted_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict emotion from a speech audio file.")
    parser.add_argument("audio_path", type=Path, help="Path to .wav/.mp3/.flac audio file.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=find_default_checkpoint(),
        help="Path to trained model checkpoint (.joblib or .pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio_path.exists():
        raise SystemExit(f"Audio file not found: {args.audio_path}")
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found at: {args.checkpoint}. Run training first.")

    emotion, probs = predict_audio_file(args.audio_path, args.checkpoint)
    print(f"\n==========================================")
    print(f"Predicted Emotion: {emotion.upper()}")
    print(f"==========================================")
    print("Confidence scores:")
    for label, score in probs.items():
        bar = "█" * int(score * 30)
        print(f"  {label:<10s}: {score * 100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
