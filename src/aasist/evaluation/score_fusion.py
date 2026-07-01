import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from compute_eer import compute_eer


def minmax_normalize(values):
    scaler = MinMaxScaler()
    return scaler.fit_transform(values.reshape(-1, 1)).reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aasist_score", required=True)
    parser.add_argument("--w2v2_score", required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--output", default="fusion_scores.txt")
    args = parser.parse_args()

    aas = pd.read_csv(args.aasist_score, sep=r"\s+", engine="python")
    wav = pd.read_csv(args.w2v2_score, sep=r"\s+", engine="python")

    df = aas.merge(wav, on="utt_id", suffixes=("_aasist", "_w2v2"))

    df["y"] = (df["label_aasist"] == "bonafide").astype(int)

    aas_score = df["score_aasist"].astype(float).values
    w2v2_score = df["score_w2v2"].astype(float).values

    aas_norm = minmax_normalize(aas_score)
    w2v2_norm = minmax_normalize(w2v2_score)

    df["fusion_score"] = args.alpha * w2v2_norm + (1.0 - args.alpha) * aas_norm

    eer, threshold = compute_eer(df["y"], df["fusion_score"])

    print(f"Alpha Wav2Vec2: {args.alpha}")
    print(f"Alpha AASIST: {1.0 - args.alpha}")
    print(f"Fusion EER: {eer:.4f}%")
    print(f"Threshold: {threshold:.6f}")

    df[["utt_id", "label_aasist", "fusion_score"]].to_csv(
        args.output,
        sep=" ",
        index=False
    )


if __name__ == "__main__":
    main()
