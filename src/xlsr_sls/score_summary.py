"""
Summarize a score file.

Expected score file format:
    <utt_id> <score>
"""

import argparse
from pathlib import Path

import pandas as pd


def summarize_score_file(score_file: Path) -> None:
    if not score_file.exists():
        raise FileNotFoundError(f"Score file not found: {score_file}")

    df = pd.read_csv(
        score_file,
        sep=r"\s+",
        header=None,
        names=["utt_id", "score"],
        engine="python",
    )

    print(f"Score file: {score_file}")
    print(f"Number of lines: {len(df)}")
    print()
    print("Score statistics:")
    print(df["score"].describe())
    print()
    print("First 5 rows:")
    print(df.head())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score_file",
        required=True,
        type=Path,
        help="Path to score file",
    )

    args = parser.parse_args()
    summarize_score_file(args.score_file)


if __name__ == "__main__":
    main()
