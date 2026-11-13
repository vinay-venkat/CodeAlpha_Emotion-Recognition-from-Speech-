"""Model architectures for Speech Emotion Recognition (PyTorch & Scikit-Learn)."""

from typing import Any, Dict, Optional

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    _BaseModule = nn.Module
except (ImportError, OSError):
    torch = None
    nn = None
    TORCH_AVAILABLE = False
    _BaseModule = object


if TORCH_AVAILABLE:
    class AttentionPooling(_BaseModule):
        """Learned additive attention pooling over temporal sequence frames."""

        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            weights = torch.softmax(self.attn(x), dim=1)
            return torch.sum(x * weights, dim=1)


    class SER_CNN(_BaseModule):
        """2D CNN operating on MFCC spectrograms `(batch, time_frames, n_features)`."""

        def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.35) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout2d(dropout * 0.5),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout2d(dropout * 0.5),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.transpose(1, 2).unsqueeze(1)
            x = self.features(x)
            return self.classifier(x)


    class SER_BiLSTM(_BaseModule):
        """Bidirectional LSTM with temporal attention over MFCC sequences."""

        def __init__(
            self,
            input_dim: int,
            num_classes: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.35,
        ) -> None:
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
            )
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.attention = AttentionPooling(hidden_dim * 2)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.proj(x)
            seq_out, _ = self.lstm(x)
            pooled = self.attention(seq_out)
            return self.classifier(pooled)


    class SER_CNN_BiLSTM(_BaseModule):
        """Hybrid 1D-CNN + Bidirectional LSTM + Attention model for speech emotion recognition."""

        def __init__(
            self,
            input_dim: int,
            num_classes: int,
            hidden_dim: int = 128,
            dropout: float = 0.35,
        ) -> None:
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout * 0.5),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout * 0.5),
            )
            self.lstm = nn.LSTM(
                input_size=256,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )
            self.attention = AttentionPooling(hidden_dim * 2)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.transpose(1, 2)
            x = self.conv_block(x)
            x = x.transpose(1, 2)
            seq_out, _ = self.lstm(x)
            pooled = self.attention(seq_out)
            return self.classifier(pooled)


def build_sklearn_model(model_type: str = "mlp", random_state: int = 42) -> Pipeline:
    """Instantiate a Scikit-Learn model pipeline with scaling."""
    key = model_type.lower().strip()
    if key in ("mlp", "deep_mlp", "neural_net"):
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=32,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=12,
            validation_fraction=0.1,
            random_state=random_state,
        )
    elif key in ("rf", "random_forest"):
        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif key in ("ensemble", "voting"):
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            random_state=random_state,
        )
        rf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        clf = VotingClassifier(
            estimators=[("mlp", mlp), ("rf", rf)],
            voting="soft",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown sklearn model type: '{model_type}' (choose 'mlp', 'rf', 'ensemble')")

    return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])


def build_model(
    model_type: str,
    input_dim: int = 120,
    num_classes: int = 7,
    dropout: float = 0.35,
) -> Any:
    """Instantiate either a PyTorch or Scikit-Learn model."""
    key = model_type.lower().strip()
    if key in ("mlp", "rf", "ensemble", "deep_mlp", "voting", "random_forest"):
        return build_sklearn_model(key)

    if not TORCH_AVAILABLE:
        raise RuntimeError(
            f"PyTorch is not available to build '{model_type}'. "
            "Use '--model mlp' or '--model ensemble' to train using Deep MLP Neural Networks, "
            "or adjust Windows Smart App Control settings to allow PyTorch DLLs."
        )

    registry = {
        "cnn": SER_CNN,
        "lstm": SER_BiLSTM,
        "cnn_lstm": SER_CNN_BiLSTM,
    }
    if key not in registry:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(registry.keys()) + ['mlp', 'rf', 'ensemble']}.")
    return registry[key](input_dim=input_dim, num_classes=num_classes, dropout=dropout)
