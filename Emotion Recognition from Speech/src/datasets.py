"""Dataset discovery and unified metadata builders for RAVDESS, TESS, and EMO-DB."""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from src.config import (
    CANONICAL_EMOTIONS,
    EMODB_EMOTION_MAP,
    RAVDESS_EMOTION_MAP,
    TESS_EMOTION_MAP,
)


def parse_ravdess_file(filepath: Path, merge_calm_to_neutral: bool = True) -> Optional[dict]:
    """Parse a RAVDESS speech wav filename like `03-01-06-01-02-01-12.wav`."""
    parts = filepath.stem.split("-")
    if len(parts) != 7:
        return None

    modality, vocal_channel, emotion_code, intensity, _stmt, _rep, actor = parts
    if modality != "03" or vocal_channel != "01":  # Keep audio-only speech
        return None

    emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
    if emotion is None:
        return None
    if emotion == "calm" and merge_calm_to_neutral:
        emotion = "neutral"

    return {
        "filepath": str(filepath.resolve()),
        "dataset": "ravdess",
        "speaker_id": f"ravdess_actor_{actor}",
        "emotion": emotion,
        "intensity": "normal" if intensity == "01" else "strong",
    }


def parse_tess_file(filepath: Path) -> Optional[dict]:
    """Parse a TESS wav filename like `OAF_back_angry.wav` or `YAF_bar_ps.wav`."""
    stem_parts = filepath.stem.lower().split("_")
    if len(stem_parts) < 2:
        return None

    speaker_token = stem_parts[0]
    if speaker_token not in {"oaf", "yaf"}:
        # Check parent folder if speaker token is encoded there
        parent_lower = filepath.parent.name.lower()
        if parent_lower.startswith("oaf"):
            speaker_token = "oaf"
        elif parent_lower.startswith("yaf"):
            speaker_token = "yaf"
        else:
            return None

    emotion_token = stem_parts[-1]
    emotion = TESS_EMOTION_MAP.get(emotion_token)
    if emotion is None:
        return None

    return {
        "filepath": str(filepath.resolve()),
        "dataset": "tess",
        "speaker_id": f"tess_{speaker_token}",
        "emotion": emotion,
        "intensity": "normal",
    }


def parse_emodb_file(filepath: Path) -> Optional[dict]:
    """Parse an EMO-DB wav filename like `03a01Fa.wav`."""
    stem = filepath.stem
    if len(stem) < 6:
        return None

    speaker_code = stem[:2]
    emotion_char = stem[5].upper()
    if not speaker_code.isdigit():
        return None

    emotion = EMODB_EMOTION_MAP.get(emotion_char)
    if emotion is None:
        return None

    return {
        "filepath": str(filepath.resolve()),
        "dataset": "emodb",
        "speaker_id": f"emodb_spk_{speaker_code}",
        "emotion": emotion,
        "intensity": "normal",
    }


def parse_custom_folder_file(filepath: Path, root: Path) -> Optional[dict]:
    """Fallback parser for folder-labeled audio: `<root>/<emotion>/<file>.wav`."""
    try:
        rel = filepath.relative_to(root)
    except ValueError:
        return None
    if len(rel.parts) < 2:
        return None

    label = rel.parts[-2].lower().strip()
    label = TESS_EMOTION_MAP.get(label, label)
    if label not in CANONICAL_EMOTIONS:
        return None

    return {
        "filepath": str(filepath.resolve()),
        "dataset": "custom",
        "speaker_id": f"custom_{filepath.stem.split('_')[0]}",
        "emotion": label,
        "intensity": "normal",
    }


def resolve_dataset_dir(raw_dir: Path, ds_name: str) -> Path:
    """Find dataset folder flexibly by name or common variations."""
    if not raw_dir.exists():
        return raw_dir

    for item in raw_dir.iterdir():
        if item.is_dir():
            name_lower = item.name.lower()
            if ds_name == "emodb" and ("emodb" in name_lower or "emo-db" in name_lower):
                return item
            if ds_name == "tess" and "tess" in name_lower:
                return item
            if ds_name == "ravdess" and "ravdess" in name_lower:
                return item
            if ds_name == name_lower:
                return item

    return raw_dir / ds_name if (raw_dir / ds_name).exists() else raw_dir


def build_metadata_dataframe(
    raw_dir: Path,
    datasets: Iterable[str] = ("ravdess", "tess", "emodb"),
    allowed_emotions: Optional[Iterable[str]] = CANONICAL_EMOTIONS,
    merge_calm_to_neutral: bool = True,
) -> pd.DataFrame:
    """Scan `raw_dir` for supported datasets and return a clean metadata DataFrame."""
    selected = {d.lower().strip() for d in datasets}
    allowed_set = set(allowed_emotions) if allowed_emotions is not None else None
    records: List[dict] = []

    dataset_parsers = {
        "ravdess": lambda p: parse_ravdess_file(p, merge_calm_to_neutral=merge_calm_to_neutral),
        "tess": parse_tess_file,
        "emodb": parse_emodb_file,
    }

    for ds_name in sorted(selected):
        search_root = resolve_dataset_dir(raw_dir, ds_name)
        wav_files = sorted(search_root.rglob("*.wav"))

        for wav_path in wav_files:
            if ds_name in dataset_parsers:
                row = dataset_parsers[ds_name](wav_path)
            elif ds_name == "custom":
                row = parse_custom_folder_file(wav_path, search_root)
            else:
                row = None

            if row is None:
                continue
            if allowed_set is not None and row["emotion"] not in allowed_set:
                continue
            records.append(row)

    df = pd.DataFrame(records)
    if not df.empty:
        # Deduplicate to prevent double-counting if dataset was extracted into nested folders
        df["_filename"] = df["filepath"].apply(lambda p: Path(p).name)
        df = df.drop_duplicates(subset=["dataset", "_filename"]).reset_index(drop=True)
        df = df.drop(columns=["_filename"])
    return df


def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    speaker_independent: bool = True,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split metadata into train, validation, and test sets without speaker leakage when possible."""
    if df.empty:
        raise ValueError("Dataset DataFrame is empty. Download datasets into data/raw/ first.")

    unique_speakers = df["speaker_id"].nunique()
    use_groups = speaker_independent and unique_speakers >= 5

    if use_groups:
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train_val_idx, test_idx = next(gss_test.split(df, df["emotion"], groups=df["speaker_id"]))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        relative_val_size = val_size / (1.0 - test_size)
        if train_val_df["speaker_id"].nunique() >= 3:
            gss_val = GroupShuffleSplit(
                n_splits=1, test_size=relative_val_size, random_state=random_seed + 1
            )
            train_idx, val_idx = next(
                gss_val.split(train_val_df, train_val_df["emotion"], groups=train_val_df["speaker_id"])
            )
            train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
            val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
            return train_df, val_df, test_df

    # Fallback: stratified split (e.g., TESS only has 2 speakers)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_val_idx, test_idx = next(sss_test.split(df, df["emotion"]))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    relative_val_size = val_size / (1.0 - test_size)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_val_size, random_state=random_seed + 1
    )
    train_idx, val_idx = next(sss_val.split(train_val_df, train_val_df["emotion"]))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    # Ensure every class in train appears sensibly; warn if missing
    missing_val = set(train_df["emotion"]) - set(val_df["emotion"])
    if missing_val:
        print(f"Warning: Validation split is missing classes {sorted(missing_val)}.")
    return train_df, val_df, test_df

