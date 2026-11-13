"""End-to-end training pipeline for Speech Emotion Recognition."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

from src.config import AudioConfig, PathConfig, TrainConfig
from src.datasets import build_metadata_dataframe, split_dataset
from src.features import (
    SequenceFeatureScaler,
    augment_waveform,
    extract_mfcc_sequence,
    extract_pooled_features,
    load_and_preprocess_audio,
)
from src.models import TORCH_AVAILABLE, build_model, build_sklearn_model

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from sklearn.utils.class_weight import compute_class_weight


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def extract_split_features(
    df: pd.DataFrame,
    audio_cfg: AudioConfig,
    desc: str,
    include_sequences: bool = False,
    waveform_aug_copies: int = 0,
    seed: int = 42,
) -> Tuple[Optional[np.ndarray], np.ndarray, List[str], int]:
    """Load audio files for a split and extract MFCC sequences + pooled acoustic vectors."""
    rng = np.random.default_rng(seed)
    sequences: List[np.ndarray] = []
    pooled_vectors: List[np.ndarray] = []
    emotions: List[str] = []
    failed_count = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=desc):
        try:
            signal = load_and_preprocess_audio(row.filepath, audio_cfg)
            pooled = extract_pooled_features(signal, audio_cfg)
            if include_sequences:
                seq = extract_mfcc_sequence(signal, audio_cfg)
                sequences.append(seq)
        except Exception as exc:
            failed_count += 1
            continue

        pooled_vectors.append(pooled)
        emotions.append(row.emotion)

        for _ in range(waveform_aug_copies):
            aug_sig = augment_waveform(signal, audio_cfg.sample_rate, rng=rng)
            pooled_vectors.append(extract_pooled_features(aug_sig, audio_cfg))
            if include_sequences:
                sequences.append(extract_mfcc_sequence(aug_sig, audio_cfg))
            emotions.append(row.emotion)

    if not pooled_vectors:
        raise RuntimeError(f"No valid audio samples extracted for {desc}.")

    seq_arr = np.stack(sequences) if include_sequences and sequences else None
    return seq_arr, np.stack(pooled_vectors), emotions, failed_count


def run_baselines(
    X_train_pooled: np.ndarray,
    y_train: np.ndarray,
    X_test_pooled: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Evaluate majority-class naive baseline and Logistic Regression baseline."""
    majority_class = int(np.bincount(y_train).argmax())
    naive_preds = np.full_like(y_test, fill_value=majority_class)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_pooled)
    X_test_sc = scaler.transform(X_test_pooled)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train_sc, y_train)
    lr_preds = clf.predict(X_test_sc)

    return {
        "naive_majority": {
            "accuracy": float(accuracy_score(y_test, naive_preds)),
            "macro_f1": float(f1_score(y_test, naive_preds, average="macro", zero_division=0)),
        },
        "logistic_regression_acoustic_stats": {
            "accuracy": float(accuracy_score(y_test, lr_preds)),
            "macro_f1": float(f1_score(y_test, lr_preds, average="macro", zero_division=0)),
        },
    }


def save_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["train_loss"], label="Train Loss", color="#2563eb")
    if history.get("val_loss"):
        axes[0].plot(history["val_loss"], label="Val Loss", color="#dc2626")
    axes[0].set_title("Training Loss Curve")
    axes[0].set_xlabel("Epoch / Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    if history.get("train_acc"):
        axes[1].plot(history["train_acc"], label="Train Acc", color="#2563eb")
    if history.get("val_acc"):
        axes[1].plot(history["val_acc"], label="Val Acc", color="#16a34a")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch / Iteration")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Held-out Test Confusion Matrix")
    ax.set_xlabel("Predicted Emotion")
    ax.set_ylabel("True Emotion")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def train_sklearn_pipeline(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    path_cfg: PathConfig,
    audio_cfg: AudioConfig,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train Deep MLP, Random Forest, or Ensemble model with Scikit-Learn."""
    pipeline = build_sklearn_model(model_type, random_state=seed)
    print(f"Training `{model_type}` classifier on {len(X_train)} samples...")

    pipeline.fit(X_train, y_train)

    train_preds = pipeline.predict(X_train)
    val_preds = pipeline.predict(X_val)
    test_preds = pipeline.predict(X_test)

    train_acc = float(accuracy_score(y_train, train_preds))
    val_acc = float(accuracy_score(y_val, val_preds))
    test_acc = float(accuracy_score(y_test, test_preds))
    test_macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(y_test, test_preds, average="weighted", zero_division=0))

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_acc": [train_acc], "val_acc": [val_acc]}
    clf_step = pipeline.named_steps.get("classifier")
    if hasattr(clf_step, "loss_curve_"):
        history["train_loss"] = [float(l) for l in clf_step.loss_curve_]
        if hasattr(clf_step, "validation_scores_") and clf_step.validation_scores_:
            history["val_acc"] = [float(s) for s in clf_step.validation_scores_]
    else:
        history["train_loss"] = [1.0 - train_acc]

    checkpoint_path = path_cfg.checkpoint_dir / "best_model.joblib"
    bundle = {
        "pipeline": pipeline,
        "model_type": model_type,
        "classes": class_names,
        "audio_config": audio_cfg.__dict__,
        "feature_dim": X_train.shape[1],
        "framework": "sklearn",
    }
    joblib.dump(bundle, checkpoint_path)

    return {
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "y_true": y_test,
        "y_pred": test_preds,
        "history": history,
        "checkpoint_path": checkpoint_path,
    }


def parse_args() -> argparse.Namespace:
    default_model = "cnn_lstm" if TORCH_AVAILABLE else "mlp"
    parser = argparse.ArgumentParser(description="Train Speech Emotion Recognition model.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ravdess", "tess", "emodb"],
        help="Datasets inside data/ to include (ravdess, tess, emodb, custom).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Custom path to raw datasets directory (defaults to data/ or data/raw/).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        choices=["mlp", "rf", "ensemble", "cnn", "lstm", "cnn_lstm"],
        help="Model architecture: 'mlp' (Deep Neural Net), 'rf' (Random Forest), 'ensemble', or PyTorch 'cnn'/'lstm'/'cnn_lstm'.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--waveform-aug-copies", type=int, default=0)
    parser.add_argument("--no-speaker-split", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_cfg = AudioConfig()
    path_cfg = PathConfig()
    path_cfg.ensure_dirs()

    # Determine raw directory: check data/ first, then data/raw/
    if args.raw_dir is not None:
        raw_dir = args.raw_dir
    elif any(path_cfg.data_dir.glob("*.wav")) or any(path_cfg.data_dir.glob("*/*.wav")) or any(path_cfg.data_dir.glob("*/*/*.wav")):
        raw_dir = path_cfg.data_dir
    else:
        raw_dir = path_cfg.raw_dir

    train_cfg = TrainConfig(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        speaker_independent=not args.no_speaker_split,
        random_seed=args.seed,
    )
    set_seed(train_cfg.random_seed)

    print(f"Scanning `{raw_dir}` for datasets: {args.datasets}")
    metadata_df = build_metadata_dataframe(raw_dir, datasets=args.datasets)
    if metadata_df.empty:
        raise SystemExit(
            f"No matching audio files found in {raw_dir}.\n"
            "Please ensure dataset folders exist inside data/."
        )

    metadata_df.to_csv(path_cfg.processed_dir / "metadata.csv", index=False)
    print(f"Total deduplicated audio files: {len(metadata_df)}")
    print("Dataset counts:\n", metadata_df["dataset"].value_counts().to_string())
    print("\nEmotion distribution:\n", metadata_df["emotion"].value_counts().to_string())

    train_df, val_df, test_df = split_dataset(
        metadata_df,
        val_size=train_cfg.val_size,
        test_size=train_cfg.test_size,
        speaker_independent=train_cfg.speaker_independent,
        random_seed=train_cfg.random_seed,
    )
    print(f"\nSplit sizes -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_df.to_csv(path_cfg.processed_dir / "train_split.csv", index=False)
    val_df.to_csv(path_cfg.processed_dir / "val_split.csv", index=False)
    test_df.to_csv(path_cfg.processed_dir / "test_split.csv", index=False)

    is_torch_model = args.model in ("cnn", "lstm", "cnn_lstm")
    if is_torch_model and not TORCH_AVAILABLE:
        print("\n[NOTE] PyTorch DLLs are restricted on this environment by Windows Smart App Control.")
        print(f"Automatically switching model from '{args.model}' to Deep Neural Network 'mlp'.")
        args.model = "mlp"
        is_torch_model = False

    _, X_train_pool, y_train_str, _ = extract_split_features(
        train_df, audio_cfg, desc="Extracting train features", include_sequences=is_torch_model,
        waveform_aug_copies=args.waveform_aug_copies, seed=train_cfg.random_seed
    )
    _, X_val_pool, y_val_str, _ = extract_split_features(
        val_df, audio_cfg, desc="Extracting val features", include_sequences=is_torch_model
    )
    _, X_test_pool, y_test_str, _ = extract_split_features(
        test_df, audio_cfg, desc="Extracting test features", include_sequences=is_torch_model
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_str)
    y_val = label_encoder.transform(y_val_str)
    y_test = label_encoder.transform(y_test_str)
    class_names = list(label_encoder.classes_)

    # Baselines
    baselines = run_baselines(X_train_pool, y_train, X_test_pool, y_test)
    print("\n--- Baseline Results (Test Set) ---")
    for name, metrics in baselines.items():
        print(f"{name}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")

    results = train_sklearn_pipeline(
        model_type=args.model,
        X_train=X_train_pool,
        y_train=y_train,
        X_val=X_val_pool,
        y_val=y_val,
        X_test=X_test_pool,
        y_test=y_test,
        class_names=class_names,
        path_cfg=path_cfg,
        audio_cfg=audio_cfg,
        seed=train_cfg.random_seed,
    )

    y_true = results["y_true"]
    y_pred = results["y_pred"]
    print("\n--- Final Test Set Evaluation ---")
    print(
        f"Model: {args.model} | Accuracy: {results['test_acc']:.4f} | "
        f"Macro F1: {results['test_macro_f1']:.4f} | Weighted F1: {results['test_weighted_f1']:.4f}"
    )
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    save_training_curves(results["history"], path_cfg.output_dir / "training_curves.png")
    save_confusion_matrix(y_true, y_pred, class_names, path_cfg.output_dir / "confusion_matrix.png")

    summary = {
        "model_type": args.model,
        "datasets": args.datasets,
        "num_samples": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "classes": class_names,
        "baselines": baselines,
        "test_metrics": {
            "accuracy": float(results["test_acc"]),
            "macro_f1": float(results["test_macro_f1"]),
            "weighted_f1": float(results["test_weighted_f1"]),
        },
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        ),
    }
    with open(path_cfg.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved model checkpoint to: {results['checkpoint_path']}")
    print(f"Saved evaluation plots & metrics to: {path_cfg.output_dir}")


if __name__ == "__main__":
    main()
