"""Standalone evaluation script for a trained Speech Emotion Recognition model."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import AudioConfig, PathConfig
from src.features import extract_pooled_features, load_and_preprocess_audio
from src.predict import find_default_checkpoint, load_predictor
from src.train import save_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SER checkpoint on test_split.csv.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=find_default_checkpoint(),
        help="Path to saved model checkpoint (.joblib or .pt).",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=PathConfig().processed_dir / "test_split.csv",
        help="CSV containing filepath and emotion columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_cfg = PathConfig()
    path_cfg.ensure_dirs()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found at {args.checkpoint}. Run `python -m src.train` first.")
    if not args.split_csv.exists():
        raise SystemExit(f"Split CSV not found at {args.split_csv}. Run `python -m src.train` first.")

    model_obj, audio_cfg, class_names, framework = load_predictor(args.checkpoint)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    df = pd.read_csv(args.split_csv)
    df = df[df["emotion"].isin(class_to_idx)].reset_index(drop=True)

    y_true_list = []
    y_pred_list = []
    skipped = 0

    print(f"Evaluating {len(df)} samples from {args.split_csv} with model: {args.checkpoint.name} ({framework})")

    if framework == "sklearn":
        features_list = []
        for row in df.itertuples():
            try:
                sig = load_and_preprocess_audio(row.filepath, audio_cfg)
                feat = extract_pooled_features(sig, audio_cfg)
                features_list.append(feat)
                y_true_list.append(class_to_idx[row.emotion])
            except Exception:
                skipped += 1
        X = np.stack(features_list)
        y_true = np.array(y_true_list)
        y_pred = model_obj.predict(X)
    else:
        import torch
        from src.features import extract_mfcc_sequence
        model, scaler, device = model_obj
        for row in df.itertuples():
            try:
                sig = load_and_preprocess_audio(row.filepath, audio_cfg)
                seq = extract_mfcc_sequence(sig, audio_cfg)
                seq_scaled = scaler.transform(seq[np.newaxis, ...])
                with torch.no_grad():
                    logits = model(torch.tensor(seq_scaled, dtype=torch.float32, device=device))
                    pred = int(torch.argmax(logits, dim=1).item())
                y_true_list.append(class_to_idx[row.emotion])
                y_pred_list.append(pred)
            except Exception:
                skipped += 1
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    print(f"\n==========================================")
    print(f"Test Accuracy   : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1-Score  : {macro_f1:.4f}")
    print(f"Weighted F1     : {weighted_f1:.4f}")
    print(f"Evaluated clips : {len(y_true)} (skipped: {skipped})")
    print(f"==========================================\n")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    out_cm = path_cfg.output_dir / "eval_confusion_matrix.png"
    save_confusion_matrix(y_true, y_pred, class_names, out_cm)
    print(f"Saved evaluation confusion matrix to: {out_cm}")


if __name__ == "__main__":
    main()
