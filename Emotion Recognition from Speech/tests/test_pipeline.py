"""Unit and integration tests for Speech Emotion Recognition pipeline."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.config import AudioConfig
from src.datasets import (
    build_metadata_dataframe,
    parse_emodb_file,
    parse_ravdess_file,
    parse_tess_file,
    split_dataset,
)
from src.features import (
    SequenceFeatureScaler,
    extract_mfcc_sequence,
    extract_pooled_features,
    load_and_preprocess_audio,
)
from src.models import TORCH_AVAILABLE, build_model


def test_filename_parsers() -> None:
    rav = parse_ravdess_file(Path("Actor_12/03-01-05-02-01-01-12.wav"))
    assert rav is not None
    assert rav["emotion"] == "angry"
    assert rav["speaker_id"] == "ravdess_actor_12"

    tess = parse_tess_file(Path("OAF_Pleasant_surprise/OAF_back_ps.wav"))
    assert tess is not None
    assert tess["emotion"] == "surprised"
    assert tess["speaker_id"] == "tess_oaf"

    emodb = parse_emodb_file(Path("wav/03a01Fa.wav"))
    assert emodb is not None
    assert emodb["emotion"] == "happy"
    assert emodb["speaker_id"] == "emodb_spk_03"


def test_audio_and_mfcc_extraction(tmp_path: Path) -> None:
    cfg = AudioConfig(sample_rate=22050, duration=2.0, n_mfcc=40, use_deltas=True)
    t = np.linspace(0, 2.0, 44100, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    wav_path = tmp_path / "sample.wav"
    sf.write(wav_path, tone, 22050)

    loaded = load_and_preprocess_audio(wav_path, cfg)
    assert loaded.shape == (cfg.target_length,)

    seq = extract_mfcc_sequence(loaded, cfg)
    assert seq.ndim == 2
    assert seq.shape[1] == cfg.n_features  # 40 * 3 = 120

    pooled = extract_pooled_features(loaded, cfg)
    assert pooled.shape == (272,)


def test_scaler_fit_transform_shapes() -> None:
    rng = np.random.default_rng(0)
    train_x = rng.normal(loc=5.0, scale=2.0, size=(12, 87, 120)).astype(np.float32)
    test_x = rng.normal(loc=5.0, scale=2.0, size=(4, 87, 120)).astype(np.float32)

    scaler = SequenceFeatureScaler()
    train_scaled = scaler.fit_transform(train_x)
    test_scaled = scaler.transform(test_x)

    assert train_scaled.shape == train_x.shape
    assert test_scaled.shape == test_x.shape
    assert np.allclose(train_scaled.mean(axis=(0, 1)), 0.0, atol=1e-4)


def test_sklearn_models() -> None:
    for model_name in ("mlp", "rf", "ensemble"):
        model = build_model(model_name)
        assert model is not None
        X = np.random.randn(20, 272)
        y = np.random.randint(0, 7, size=20)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 20


def test_pytorch_models() -> None:
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available on this host environment.")

    import torch
    batch_x = torch.randn(4, 130, 120)
    batch_y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    for model_type in ("cnn", "lstm", "cnn_lstm"):
        model = build_model(model_type, input_dim=120, num_classes=7, dropout=0.2)
        logits = model(batch_x)
        assert logits.shape == (4, 7)
        loss = torch.nn.functional.cross_entropy(logits, batch_y)
        loss.backward()


def test_metadata_and_speaker_split(tmp_path: Path) -> None:
    rav_dir = tmp_path / "ravdess"
    rav_dir.mkdir(parents=True)
    tone = np.zeros(22050, dtype=np.float32)

    for actor in range(1, 7):
        for emo in ("01", "03", "04", "05"):
            fp = rav_dir / f"03-01-{emo}-01-01-01-{actor:02d}.wav"
            sf.write(fp, tone, 22050)

    df = build_metadata_dataframe(tmp_path, datasets=["ravdess"])
    assert len(df) == 24

    train_df, val_df, test_df = split_dataset(df, speaker_independent=True, random_seed=42)
    assert len(train_df) + len(val_df) + len(test_df) == 24
    assert set(train_df["speaker_id"]).isdisjoint(set(test_df["speaker_id"]))
