"""Central configuration for Speech Emotion Recognition (SER)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AudioConfig:
    """Audio loading and MFCC extraction settings."""

    sample_rate: int = 22050
    duration: float = 3.0  # seconds
    n_mfcc: int = 40
    n_fft: int = 2048
    hop_length: int = 512
    use_deltas: bool = True  # Include delta and delta-delta MFCCs (3x features)
    trim_silence: bool = True
    top_db: int = 25

    @property
    def target_length(self) -> int:
        return int(self.sample_rate * self.duration)

    @property
    def n_features(self) -> int:
        return self.n_mfcc * 3 if self.use_deltas else self.n_mfcc


@dataclass
class TrainConfig:
    """Training and model hyperparameters."""

    model_type: str = "cnn_lstm"  # Options: "cnn", "lstm", "cnn_lstm"
    batch_size: int = 32
    epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.35
    patience: int = 10
    val_size: float = 0.15
    test_size: float = 0.15
    random_seed: int = 42
    augment_train: bool = True
    speaker_independent: bool = True
    num_workers: int = 0  # Windows-safe default for PyTorch DataLoader


@dataclass
class PathConfig:
    """Standard workspace paths."""

    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    checkpoint_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoints")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.raw_dir, self.processed_dir, self.checkpoint_dir, self.output_dir):
            path.mkdir(parents=True, exist_ok=True)


# Canonical emotion set shared across RAVDESS, TESS, and EMO-DB
CANONICAL_EMOTIONS: List[str] = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

# RAVDESS filename emotion codes (3rd field)
RAVDESS_EMOTION_MAP: Dict[str, str] = {
    "01": "neutral",
    "02": "calm",       # Optionally merged into neutral or dropped if using 7 emotions
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# TESS filename / folder emotion tokens
TESS_EMOTION_MAP: Dict[str, str] = {
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fearful",
    "fearful": "fearful",
    "disgust": "disgust",
    "ps": "surprised",
    "surprised": "surprised",
    "pleasant_surprise": "surprised",
    "pleasant_surprised": "surprised",
}

# EMO-DB filename 6th-character emotion codes
EMODB_EMOTION_MAP: Dict[str, str] = {
    "N": "neutral",
    "F": "happy",
    "T": "sad",
    "W": "angry",
    "A": "fearful",
    "E": "disgust",
    "L": "boredom",  # Not in standard 7-class set; filtered unless included
}

