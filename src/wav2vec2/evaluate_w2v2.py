import os
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .w2v2_model import Wav2Vec2DeepfakeDetector
from .data_utils import (
    WavDataset,
    load_asv2021_metadata,
    load_add2023_labels,
)
from .metrics import compute_eer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["asv2021_la", "asv2021_df", "add2023"],
    )

    parser.add_argument("--model_path", required=True)

    parser.add_argument("--meta_path", default=None)
    parser.add_argument("--label_path", default=None)
    parser.add_argument("--audio_dirs", required=True)

    parser.add_argument("--score_path", required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    audio_dirs = args.audio_dirs.split(",")

    if args.dataset in ["asv2021_la", "asv2021_df"]:
        if args.meta_path is None:
            raise ValueError("--meta_path is required for ASVspoof2021 datasets.")

        df = load_asv2021_metadata(
            meta_path=args.meta_path,
            audio_dirs=audio_dirs,
            file_col=1,
            label_col=5,
            subset_col=7,
        )

    elif args.dataset == "add2023":
        if args.label_path is None:
            raise ValueError("--label_path is required for ADD2023.")

        df = load_add2023_labels(
            label_path=args.label_path,
            audio_dirs=audio_dirs,
        )

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    loader = DataLoader(
        WavDataset(df, train=False, aug_level="none"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = Wav2Vec2DeepfakeDetector().to(device)

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    for name, param in model.named_parameters():
        if "layer_weights" in name:
            print("CHECK:", name, param.shape)

    all_file_ids = []
    all_labels = []
    all_label_strs = []
    all_scores = []
    all_subsets = []
    all_ok_audio = []

    label_map = {
        0: "bonafide",
        1: "spoof",
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {args.dataset}"):
            x, y, file_ids, subsets, ok_audio = batch

            x = x.to(device, non_blocking=True)

            logits = model(x)

            # Final score used in this thesis:
            # spoof posterior probability after softmax.
            prob_spoof = torch.softmax(logits, dim=1)[:, 1]

            y_list = y.numpy().tolist()
            ok_list = ok_audio.numpy().tolist()

            all_file_ids.extend(list(file_ids))
            all_labels.extend(y_list)
            all_label_strs.extend([label_map[int(v)] for v in y_list])
            all_scores.extend(prob_spoof.detach().cpu().numpy().tolist())
            all_subsets.extend(list(subsets))
            all_ok_audio.extend(ok_list)

    result = pd.DataFrame({
        "file_id": all_file_ids,
        "label": all_labels,
        "label_str": all_label_strs,
        "score": all_scores,
        "subset": all_subsets,
        "ok_audio": all_ok_audio,
    })

    os.makedirs(os.path.dirname(args.score_path) or ".", exist_ok=True)
    result.to_csv(args.score_path, sep=" ", index=False)

    bad_path = args.score_path.replace(".txt", "_bad_audio.txt")
    bad = result[result["ok_audio"] == 0][["file_id"]]
    bad.to_csv(bad_path, index=False, header=False)

    print("\nSaved scores to:", args.score_path)
    print("Saved bad audio list to:", bad_path)
    print("Total scored:", len(result))
    print("Bad audio:", len(bad))

    print("\nScore head:")
    print(result.head())

    print("\nLabel count:")
    print(result["label_str"].value_counts())

    print("\nSubset count:")
    print(result["subset"].value_counts())

    print("\nAudio OK count:")
    print(result["ok_audio"].value_counts())

    print("\nScore statistics by label:")
    print(result.groupby("label_str")["score"].describe())

    print("\nScore statistics by subset and label:")
    print(result.groupby(["subset", "label_str"])["score"].describe())

    print("\n========== EER RESULTS ==========")

    full_eer = compute_eer(result["label"].values, result["score"].values)

    if full_eer is None:
        print("FULL EER: cannot compute.")
    else:
        print(f"FULL EER: {full_eer:.6f}% | N={len(result)}")

    for subset in ["eval", "progress", "hidden", "add2023"]:
        sub = result[result["subset"].astype(str).str.lower() == subset]

        if len(sub) == 0:
            continue

        eer = compute_eer(sub["label"].values, sub["score"].values)

        if eer is None:
            print(f"{subset.upper()} EER: cannot compute | N={len(sub)}")
        else:
            print(f"{subset.upper()} EER: {eer:.6f}% | N={len(sub)}")

    print("=================================")


if __name__ == "__main__":
    main()
