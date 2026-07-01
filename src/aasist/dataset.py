import os
import torch
from torch.utils.data import Dataset

from src.aasist.utils.audio_utils import load_audio, pad_or_truncate


class AASISTInferenceDataset(Dataset):
    """
    Dataset class for AASIST inference.

    This dataset loads audio files, resamples them to 16 kHz,
    and converts each utterance to a fixed length of 64,600 samples.
    """

    def __init__(
        self,
        metadata,
        audio_dir,
        file_col="utt_id",
        label_col=None,
        extension=".flac",
        sample_rate=16000,
        input_length=64600
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.file_col = file_col
        self.label_col = label_col
        self.extension = extension
        self.sample_rate = sample_rate
        self.input_length = input_length

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        utt_id = row[self.file_col]
        audio_path = os.path.join(self.audio_dir, utt_id + self.extension)

        audio = load_audio(audio_path, target_sr=self.sample_rate)
        audio = pad_or_truncate(audio, target_length=self.input_length)

        audio = torch.tensor(audio, dtype=torch.float32)

        if self.label_col is not None:
            label = row[self.label_col]
            return utt_id, audio, label

        return utt_id, audio
