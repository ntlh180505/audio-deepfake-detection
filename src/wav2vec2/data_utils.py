import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio_utils import (
    read_audio_robust,
    crop_or_pad,
    augment_original,
    augment_light,
)


LABEL_MAP = {
    "bonafide": 0,
    "spoof": 1,
    "genuine": 0,
    "fake": 1,
    "real": 0,
}


def find_audio_path(file_id, audio_dirs, extensions=(".flac", ".wav", ".wave")):
    """
    Search for an audio file in one or multiple directories.
    """
    candidates = [file_id]

    if not file_id.lower().endswith(extensions):
        for ext in extensions:
            candidates.append(file_id + ext)

    for audio_dir in audio_dirs:
        audio_dir = Path(audio_dir)

        for candidate in candidates:
            path = audio_dir / candidate
            if path.exists():
                return str(path)

    return None


def load_asv2019_protocol(protocol_path, audio_dir):
    """
    Load ASVspoof2019 LA train/dev/eval protocol.

    Expected format:
        speaker_id file_id system_id attack_id label

    label:
        bonafide / spoof
    """
    rows = []

    with open(protocol_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split()

            file_id = parts[1]
            label_str = parts[-1].lower()
            label = LABEL_MAP[label_str]

            path = find_audio_path(file_id, [audio_dir])

            rows.append({
                "file_id": file_id,
                "path": path,
                "label": label,
                "label_str": "bonafide" if label == 0 else "spoof",
                "subset": "asv2019",
            })

    df = pd.DataFrame(rows)
    df = df[df["path"].notna()].reset_index(drop=True)

    return df


def load_asv2021_metadata(meta_path, audio_dirs, file_col=1, label_col=5, subset_col=7):
    """
    Load ASVspoof2021 LA/DF metadata.

    For the metadata used in this thesis:
        col1 = file_id
        col5 = bonafide/spoof
        col7 = subset: eval/progress/hidden
    """
    raw = pd.read_csv(meta_path, sep=r"\s+", header=None, dtype=str)

    df = pd.DataFrame()
    df["file_id"] = raw[file_col].astype(str).str.replace(".flac", "", regex=False)
    df["label_str"] = raw[label_col].astype(str).str.lower()
    df["subset"] = raw[subset_col].astype(str).str.lower()

    df["label"] = df["label_str"].map(LABEL_MAP)

    if df["label"].isna().any():
        bad = df[df["label"].isna()].head(20)
        raise ValueError(f"Label mapping failed. Examples:\n{bad}")

    df["label"] = df["label"].astype(int)

    audio_dirs = [x.strip() for x in audio_dirs if x.strip()]
    df["path"] = df["file_id"].apply(
        lambda x: find_audio_path(x, audio_dirs)
    )

    total = len(df)
    missing = df["path"].isna().sum()

    df = df[df["path"].notna()].reset_index(drop=True)

    print("Metadata:", meta_path)
    print("Total metadata:", total)
    print("Missing audio:", missing)
    print("Existing audio:", len(df))
    print("\nLabel count:")
    print(df["label_str"].value_counts())
    print("\nSubset count:")
    print(df["subset"].value_counts())

    return df


def load_add2023_labels(label_path, audio_dirs):
    """
    Load ADD2023 Track 1.2 label file.

    Expected format:
        filename label

    label:
        genuine / fake
    """
    rows = []

    with open(label_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split()

            if len(parts) < 2:
                continue

            fname = parts[0]
            label_str_raw = parts[1].lower()

            label = LABEL_MAP[label_str_raw]
            label_str = "bonafide" if label == 0 else "spoof"

            path = find_audio_path(fname, audio_dirs)

            rows.append({
                "file_id": fname,
                "path": path,
                "label": label,
                "label_str": label_str,
                "subset": "add2023",
            })

    df = pd.DataFrame(rows)

    total = len(df)
    missing = df["path"].isna().sum()

    df = df[df["path"].notna()].reset_index(drop=True)

    print("Label:", label_path)
    print("Total:", total)
    print("Missing audio:", missing)
    print("Existing audio:", len(df))
    print("\nLabel count:")
    print(df["label_str"].value_counts())

    return df


class WavDataset(Dataset):
    """
    Dataset used for both training and inference.

    During training:
        - random crop for long audio
        - optional augmentation

    During evaluation:
        - first 4 seconds are used
        - no augmentation
    """
    def __init__(
        self,
        df,
        train=False,
        aug_level="none",
        max_len=16000 * 4,
    ):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.aug_level = aug_level
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio = read_audio_robust(row["path"], target_sr=16000)
        ok_audio = 1

        if audio is None:
            audio = np.zeros(self.max_len, dtype=np.float32)
            ok_audio = 0

        audio = crop_or_pad(audio, max_len=self.max_len, train=self.train)

        if self.train:
            if self.aug_level == "original":
                audio = augment_original(audio)
            elif self.aug_level == "light":
                audio = augment_light(audio)

        audio = np.clip(audio, -1.0, 1.0)

        return (
            torch.tensor(audio).float(),
            torch.tensor(row["label"]).long(),
            row["file_id"],
            row["subset"],
            torch.tensor(ok_audio).long(),
        )
