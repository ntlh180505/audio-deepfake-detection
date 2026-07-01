import numpy as np
import soundfile as sf
import librosa


def read_audio_robust(path, target_sr=16000):
    """
    Robust audio loader for .flac and .wav files.

    The loader first tries soundfile. If it fails, it falls back to librosa.
    The output is always mono, float32, and resampled to target_sr.

    Returns:
        audio: np.ndarray or None
    """
    try:
        try:
            audio, sr = sf.read(str(path))

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

        except Exception:
            audio, sr = librosa.load(str(path), sr=None, mono=True)

        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if len(audio) == 0:
            return None

        return audio

    except Exception:
        return None


def crop_or_pad(audio, max_len=16000 * 4, train=False):
    """
    Crop or pad audio to a fixed length.

    During training, long utterances are randomly cropped.
    During evaluation, the first max_len samples are used.
    """
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    elif len(audio) > max_len:
        if train:
            start = np.random.randint(0, len(audio) - max_len + 1)
            audio = audio[start:start + max_len]
        else:
            audio = audio[:max_len]

    return audio.astype(np.float32)


def add_snr_noise(audio, snr_db_min=15, snr_db_max=30):
    snr_db = np.random.uniform(snr_db_min, snr_db_max)
    noise = np.random.randn(len(audio)).astype(np.float32)

    signal_power = np.mean(audio ** 2) + 1e-8
    noise_power = np.mean(noise ** 2) + 1e-8

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)

    return audio + noise


def random_gain(audio, min_gain=0.8, max_gain=1.2):
    gain = np.random.uniform(min_gain, max_gain)
    return audio * gain


def time_mask(audio, max_mask_ms=250, sr=16000):
    max_mask_len = int(sr * max_mask_ms / 1000)

    if len(audio) <= max_mask_len:
        return audio

    mask_len = np.random.randint(400, max_mask_len)
    start = np.random.randint(0, len(audio) - mask_len)

    audio = audio.copy()
    audio[start:start + mask_len] = 0.0

    return audio


def augment_original(audio):
    """
    Original lightweight augmentation used in the baseline run:
    - Gaussian noise
    - amplitude scaling
    - short zero masking
    """
    if np.random.rand() < 0.5:
        audio = audio + 0.002 * np.random.randn(len(audio)).astype(np.float32)

    if np.random.rand() < 0.3:
        audio = audio * np.random.uniform(0.9, 1.1)

    if np.random.rand() < 0.3:
        start = np.random.randint(0, len(audio) // 2)
        audio = audio.copy()
        audio[start:start + 2000] = 0.0

    return audio


def augment_light(audio):
    """
    Light augmentation variant:
    - SNR-controlled additive noise
    - amplitude scaling
    - time masking
    """
    if np.random.rand() < 0.5:
        audio = add_snr_noise(audio, 15, 30)

    if np.random.rand() < 0.5:
        audio = random_gain(audio, 0.8, 1.2)

    if np.random.rand() < 0.3:
        audio = time_mask(audio, max_mask_ms=250)

    return audio
