import numpy as np
import librosa

from .config import LogMelConfig


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """
    Load audio as mono waveform and resample to the target sampling rate.

    The same loader is used for:
    - ASVspoof2019 LA: .flac
    - ASVspoof2021 LA/DF: .flac
    - ADD2023: .wav
    """
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def crop_or_pad(audio: np.ndarray, max_len: int) -> np.ndarray:
    """
    Normalize waveform length.

    If the utterance is shorter than max_len, zero-padding is applied.
    If it is longer, the utterance is truncated from the beginning.
    """
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]

    return audio.astype(np.float32)


def apply_waveform_augmentation(audio: np.ndarray) -> np.ndarray:
    """
    Light waveform-level augmentation used only during training.

    Augmentations:
    - Gaussian noise
    - amplitude scaling
    - small temporal shift
    """
    if np.random.rand() < 0.5:
        audio = audio + 0.002 * np.random.randn(len(audio))

    if np.random.rand() < 0.3:
        audio = audio * np.random.uniform(0.9, 1.1)

    if np.random.rand() < 0.3:
        audio = np.roll(audio, np.random.randint(0, 1000))

    return audio.astype(np.float32)


def waveform_to_logmel(audio: np.ndarray, config: LogMelConfig) -> np.ndarray:
    """
    Convert waveform to normalized Log-Mel spectrogram.

    Steps:
    1. Power Mel spectrogram
    2. Log compression using librosa.power_to_db(ref=np.max)
    3. Per-utterance z-score normalization
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)

    return mel.astype(np.float32)
