import numpy as np
import soundfile as sf
import librosa


def load_audio(path, target_sr=16000):
    """
    Load an audio file and resample it to the target sampling rate.

    Parameters
    ----------
    path : str
        Path to the audio file.
    target_sr : int
        Target sampling rate.

    Returns
    -------
    np.ndarray
        Audio waveform.
    """
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)

    return audio.astype(np.float32)


def pad_or_truncate(audio, target_length=64600):
    """
    Convert an audio waveform to a fixed length.

    If the audio is longer than target_length, it is truncated.
    If it is shorter, it is repeated until the required length is reached.

    Parameters
    ----------
    audio : np.ndarray
        Input waveform.
    target_length : int
        Target number of samples.

    Returns
    -------
    np.ndarray
        Fixed-length waveform.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) >= target_length:
        return audio[:target_length]

    repeat_factor = int(np.ceil(target_length / len(audio)))
    audio = np.tile(audio, repeat_factor)

    return audio[:target_length]
