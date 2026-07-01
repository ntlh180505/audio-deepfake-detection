import torch.nn as nn
import torchvision.models as models


def build_logmel_resnet18(num_classes: int = 2, dropout: float = 0.3):
    """
    Build ResNet18 for single-channel Log-Mel spectrogram input.

    Original ResNet18 expects RGB input with 3 channels.
    Since Log-Mel spectrogram is a single-channel feature map,
    the first convolution is modified from 3 input channels to 1.

    Output:
    - 2 logits: bonafide and spoof
    """
    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, num_classes),
    )

    return model
