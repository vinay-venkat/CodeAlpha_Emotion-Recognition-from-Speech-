"""Speech signal processing, data augmentation, and MFCC feature extraction."""

from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np

from src.config import AudioConfig


def load_and_preprocess_audio(
    filepath: Union[str, Path],
    config: AudioConfig,
) -> np.ndarray:
    """Load mono audio, optionally trim silence, and pad/crop to fixed duration."""
    signal, _ = librosa.load(str(filepath), sr=config.sample_rate, mono=True)

    if signal.size == 0:
        return np.zeros(config.target_length, dtype=np.float32)

    if config.trim_silence:
        trimmed, _ = librosa.effects.trim(signal, top_db=config.top_db)
        if trimmed.size > 0:
            signal = trimmed

    # Peak normalize to prevent volume bias across microphones/datasets
    max_amp = np.max(np.abs(signal))
    if max_amp > 1e-6:
        signal = signal / max_amp

    target_len = config.target_length
    if len(signal) > target_len:
        # Center-crop longest active speech segment
        start = (len(signal) - target_len) // 2
        signal = signal[start : start + target_len]
    elif len(signal) < target_len:
        pad_total = target_len - len(signal)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        signal = np.pad(signal, (pad_left, pad_right), mode="constant")

    return signal.astype(np.float32)


def augment_waveform(
    signal: np.ndarray,
    sample_rate: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply lightweight speech-preserving augmentations (noise, shift, pitch/stretch)."""
    rng = rng or np.random.default_rng()
    augmented = signal.copy()

    # 1. Random time shift (up to +/- 10% of clip)
    if rng.random() < 0.5:
        shift_max = int(0.1 * len(augmented))
        shift = int(rng.integers(-shift_max, shift_max + 1))
        augmented = np.roll(augmented, shift)

    # 2. Additive white Gaussian noise
    if rng.random() < 0.5:
        noise_amp = rng.uniform(0.002, 0.015) * np.max(np.abs(augmented) + 1e-6)
        augmented = augmented + noise_amp * rng.normal(size=len(augmented))

    # 3. Mild pitch shift (+/- 2 semitones)
    if rng.random() < 0.35:
        n_steps = float(rng.uniform(-2.0, 2.0))
        augmented = librosa.effects.pitch_shift(augmented, sr=sample_rate, n_steps=n_steps)

    # 4. Mild speed/time stretch (0.9x to 1.1x), preserving target length
    if rng.random() < 0.35:
        rate = float(rng.uniform(0.9, 1.1))
        stretched = librosa.effects.time_stretch(augmented, rate=rate)
        if len(stretched) > len(signal):
            augmented = stretched[: len(signal)]
        else:
            augmented = np.pad(stretched, (0, len(signal) - len(stretched)), mode="constant")

    return augmented.astype(np.float32)


def extract_mfcc_sequence(
    signal: np.ndarray,
    config: AudioConfig,
) -> np.ndarray:
    """Extract time-series MFCC (+ delta + delta-delta) matrix of shape `(time_frames, n_features)`."""
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )

    if config.use_deltas:
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta, delta2])
    else:
        features = mfcc

    # Transpose to (time_frames, n_features) for sequence/CNN compatibility
    return features.T.astype(np.float32)


def extract_pooled_features(signal: np.ndarray, config: AudioConfig) -> np.ndarray:
    """Extract comprehensive 272-dimensional acoustic feature vector (MFCCs, Deltas, Chroma, Prosody)."""
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(
        y=signal, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length
    )
    zcr = librosa.feature.zero_crossing_rate(signal, hop_length=config.hop_length)
    rms = librosa.feature.rms(y=signal, hop_length=config.hop_length)
    centroid = librosa.feature.spectral_centroid(
        y=signal, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=signal, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length
    )

    feature_maps = (mfcc, delta1, delta2, chroma, zcr, rms, centroid, rolloff)
    means = [np.mean(m, axis=1) for m in feature_maps]
    stds = [np.std(m, axis=1) for m in feature_maps]
    return np.concatenate(means + stds).astype(np.float32)


class SequenceFeatureScaler:
    """Feature-wise standardizer fitted strictly on training sequences to prevent data leakage."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, sequences: np.ndarray) -> "SequenceFeatureScaler":
        """Fit mean and std over `(N, T, F)` across batch and time axes."""
        self.mean_ = np.mean(sequences, axis=(0, 1), keepdims=True).astype(np.float32)
        std = np.std(sequences, axis=(0, 1), keepdims=True).astype(np.float32)
        self.std_ = np.where(std < 1e-6, 1.0, std)
        return self

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("SequenceFeatureScaler must be fitted before transform().")
        return ((sequences - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, sequences: np.ndarray) -> np.ndarray:
        return self.fit(sequences).transform(sequences)

    def state_dict(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted.")
        return self.mean_, self.std_

    @classmethod
    def from_state(cls, mean: np.ndarray, std: np.ndarray) -> "SequenceFeatureScaler":
        scaler = cls()
        scaler.mean_ = np.asarray(mean, dtype=np.float32)
        scaler.std_ = np.asarray(std, dtype=np.float32)
        return scaler

