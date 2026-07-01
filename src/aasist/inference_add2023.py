"""
Inference script for AASIST on ADD2023 Track 1.2 TestR1.

This script is a simplified template showing the inference pipeline used in this thesis.
The full AASIST model definition follows the official AASIST implementation.
"""

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.aasist.data.dataset import AASISTInferenceDataset


def load_add2023_labels(label_path):
    """
    Load ADD2023 label file.

    Expected format:
        filename label

    Example:
        ADD2023_T1.2R1_E_00000000.wav fake
        ADD2023_T1.2R1_E_00000001.wav genuine
    """
    df = pd.read_csv(
        label_path,
        sep=r"\s+",
        header=None,
        names=["fname", "label"]
    )

    df["utt_id"] = df["fname"].str.replace(".wav", "", regex=False)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--label_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="add2023_aasist_scores.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    metadata = load_add2023_labels(args.label_path)

    dataset = AASISTInferenceDataset(
        metadata=metadata,
        audio_dir=args.audio_dir,
        file_col="utt_id",
        label_col="label",
        extension=".wav",
        sample_rate=16000,
        input_length=64600
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    # ------------------------------------------------------------------
    # Load AASIST model here.
    # The model definition follows the official AASIST implementation.
    #
    # Example:
    # from src.aasist.models.aasist_model import Model
    # model = Model(config["model_config"]).to(device)
    # model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    # model.eval()
    # ------------------------------------------------------------------

    results = []

    with torch.no_grad():
        for utt_ids, audio, labels in loader:
            audio = audio.to(device)

            # logits = model(audio)
            # score = logits[:, 1] or logits[:, 0], depending on score convention

            # Placeholder for repository template
            raise NotImplementedError(
                "Please connect this script to the official AASIST model definition."
            )

    out_df = pd.DataFrame(results, columns=["utt_id", "label", "score"])
    out_df.to_csv(args.output, sep=" ", index=False)


if __name__ == "__main__":
    main()
