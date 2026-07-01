"""
Prepare ADD2023 Track 1.2 R1 protocol file for XLS-R + SLSClassifier inference.

Input label file format:
    <utt_id.wav> <label>

Example:
    ADD2023_T1.2R1_E_00000000.wav genuine
    ADD2023_T1.2R1_E_00000001.wav fake

Output protocol file format:
    <utt_id.wav>

This matches the Dataset_in_the_wild_eval class used by the original
XLS-R + SLSClassifier implementation.
"""

import argparse
from pathlib import Path


def prepare_protocol(label_file: Path, output_file: Path) -> None:
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with label_file.open("r", encoding="utf-8") as fin, output_file.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            utt_id = parts[0]
            fout.write(f"{utt_id}\n")
            count += 1

    print(f"Protocol saved to: {output_file}")
    print(f"Number of utterances: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_file",
        required=True,
        type=Path,
        help="Path to ADD2023 label.txt",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        type=Path,
        help="Output protocol file",
    )

    args = parser.parse_args()
    prepare_protocol(args.label_file, args.output_file)


if __name__ == "__main__":
    main()
