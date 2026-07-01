"""
AASIST model wrapper.

The full architecture follows the official AASIST implementation:

Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
Spectro-Temporal Graph Attention Networks", Interspeech 2022.

In this thesis, the official implementation was used for training and inference.
This file is provided as a wrapper placeholder to indicate where the model
definition should be placed when reproducing the experiments.
"""


class AASISTModelWrapper:
    def __init__(self, config):
        self.config = config

    def build(self):
        raise NotImplementedError(
            "Please use the official AASIST model definition from the original repository."
        )
