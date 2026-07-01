import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class AttentivePooling(nn.Module):
    """
    Attentive statistical pooling.

    Input:
        x: [B, T, D]

    Output:
        pooled: [B, 2D]
        concatenation of weighted mean and weighted standard deviation.
    """
    def __init__(self, dim):
        super().__init__()

        self.att = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.att(x), dim=1)

        mean = torch.sum(weights * x, dim=1)

        std = torch.sqrt(
            torch.sum(weights * (x - mean.unsqueeze(1)) ** 2, dim=1) + 1e-6
        )

        return torch.cat([mean, std], dim=1)


class Wav2Vec2DeepfakeDetector(nn.Module):
    """
    Wav2Vec2-based audio deepfake detector.

    Architecture:
        Raw waveform
        -> Wav2Vec2 base
        -> 12 transformer hidden states
        -> layer-wise weighted aggregation
        -> attentive statistical pooling
        -> FC classifier
        -> 2 logits: bonafide / spoof
    """
    def __init__(self, pretrained_name="facebook/wav2vec2-base"):
        super().__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_name)

        # Freeze all parameters except the last two transformer layers.
        for name, param in self.wav2vec.named_parameters():
            if "encoder.layers.10" not in name and "encoder.layers.11" not in name:
                param.requires_grad = False

        # Wav2Vec2-base has 12 transformer layers.
        self.layer_weights = nn.Parameter(torch.ones(12))

        self.pool = AttentivePooling(768)

        self.fc = nn.Sequential(
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        """
        x:
            waveform tensor [B, samples]

        returns:
            logits [B, 2]
        """
        out = self.wav2vec(x, output_hidden_states=True)

        # hidden_states[0] is the feature projection output.
        # hidden_states[1:] are the 12 transformer layer outputs.
        hidden = torch.stack(out.hidden_states[1:], dim=0)

        weights = torch.softmax(self.layer_weights, dim=0)

        x = torch.sum(
            weights[:, None, None, None] * hidden,
            dim=0
        )

        x = self.pool(x)

        logits = self.fc(x)

        return logits
