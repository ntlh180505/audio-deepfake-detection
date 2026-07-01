import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T

from .config import LogMelConfig
from .features import (
    load_audio,
    crop_or_pad,
    apply_waveform_augmentation,
    waveform_to_logmel,
)


def build_audio_map(audio_dirs, extensions=(".flac", ".wav")):
    """
    Scan one or multiple audio directories and map file_id to absolute path.

    Example:
    DF_E_2000011 -> /path/to/DF_E_2000011.flac
    ADD2023_T1.2R1_E_00000000 -> /path/to/ADD2023_T1.2R1_E_00000000.wav
    """
    if isinstance(audio_dirs, str):
        audio_dirs = [audio_dirs]

    audio_map = {}

    for audio_dir in audio_dirs:
        if not os.path.exists(audio_dir):
            print(f"[WARNING] Audio directory not found: {audio_dir}")
            continue

        for root, _, files in os.walk(audio_dir):
            for fname in files:
                if fname.lower().endswith(extensions):
                    file_id = os.path.splitext(fname)[0]
                    audio_map[file_id] = os.path.join(root, fname)

    return audio_map


def attach_audio_paths(df, audio_dirs, file_col="file", extensions=(".flac", ".wav")):
    audio_map = build_audio_map(audio_dirs, extensions=extensions)

    df = df.copy()
    df["file_key"] = df[file_col].apply(
        lambda x: os.path.splitext(os.path.basename(str(x)))[0]
    )
    df["path"] = df["file_key"].map(audio_map)

    return df


def load_asv2019_protocol(path):
    """
    ASVspoof2019 LA protocol format:
    speaker file env attack label
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)

    df.columns = ["speaker", "file", "env", "attack", "label_str"]

    df["label"] = df["label_str"].map({
        "bonafide": 0,
        "spoof": 1,
    })

    return df


def load_asv2021_la_protocol(path):
    """
    ASVspoof2021 LA CM trial_metadata format:
    speaker file codec transmission attack label trim subset
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)

    df.columns = [
        "speaker",
        "file",
        "codec",
        "transmission",
        "attack",
        "label_str",
        "trim",
        "subset",
    ]

    df["label"] = df["label_str"].map({
        "bonafide": 0,
        "spoof": 1,
    })

    return df


def load_asv2021_df_protocol(path):
    """
    ASVspoof2021 DF CM trial_metadata format with 13 columns.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)

    df.columns = [
        "speaker",
        "file",
        "codec",
        "source",
        "attack",
        "label_str",
        "trim",
        "subset",
        "vocoder",
        "task",
        "team",
        "gender",
        "language",
    ]

    df["label"] = df["label_str"].map({
        "bonafide": 0,
        "spoof": 1,
    })

    return df


def load_add2023_protocol(path):
    """
    Generic ADD2023 label loader.

    Expected examples:
    file.wav genuine
    file.wav fake

    or:
    file_id genuine
    file_id fake
    """
    df_raw = pd.read_csv(path, sep=r"\s+", header=None)

    label_map = {
        "genuine": 0,
        "real": 0,
        "bonafide": 0,
        "human": 0,
        "fake": 1,
        "spoof": 1,
        "synthetic": 1,
    }

    label_col = None
    for col in df_raw.columns:
        values = set(df_raw[col].astype(str).str.lower().unique())
        if len(values.intersection(set(label_map.keys()))) > 0:
            label_col = col
            break

    if label_col is None:
        raise RuntimeError("Cannot find label column in ADD2023 protocol.")

    file_col = 0 if label_col != 0 else 1

    df = pd.DataFrame()
    df["file"] = df_raw[file_col].astype(str)
    df["label_str"] = df_raw[label_col].astype(str).str.lower()
    df["label"] = df["label_str"].map(label_map)

    if df["label"].isna().any():
        print(df["label_str"].value_counts())
        raise RuntimeError("Some labels cannot be mapped.")

    return df


class LogMelDataset(Dataset):
    """
    Dataset for Log-Mel ResNet18.

    During training:
    - waveform augmentation
    - SpecAugment

    During evaluation:
    - no augmentation
    """

    def __init__(self, df, train=False, config=None):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.config = config if config is not None else LogMelConfig()

        self.freq_mask = T.FrequencyMasking(20)
        self.time_mask = T.TimeMasking(50)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio = load_audio(row["path"], sample_rate=self.config.sample_rate)
        audio = crop_or_pad(audio, self.config.max_len)

        if self.train:
            audio = apply_waveform_augmentation(audio)

        mel = waveform_to_logmel(audio, self.config)

        mel = torch.tensor(mel).float().unsqueeze(0)

        if self.train:
            if torch.rand(1).item() < 0.5:
                mel = self.freq_mask(mel)

            if torch.rand(1).item() < 0.5:
                mel = self.time_mask(mel)

        label = torch.tensor(int(row["label"])).long()

        return mel, label, row["file"]
