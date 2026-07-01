from dataclasses import dataclass


@dataclass
class LogMelConfig:
    """
    Configuration for Log-Mel spectrogram extraction.

    This configuration matches the implementation used in the thesis:
    - 16 kHz audio
    - fixed 4-second input
    - 128 Mel filters
    - n_fft = 1024
    - hop_length = 256
    - per-utterance z-score normalization
    """

    sample_rate: int = 16000
    duration: float = 4.0
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    dropout: float = 0.3

    @property
    def max_len(self) -> int:
        return int(self.sample_rate * self.duration)
