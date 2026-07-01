"""
Merge ASVspoof2021 DF score files generated from different dataset parts.

Expected input files:
    scores_DF_epoch7_part00.txt
    scores_DF_epoch7_part01.txt
    scores_DF_epoch7_part02.txt
    scores_DF_epoch7_part03.txt

Expected output:
    scores_DF_epoch7_2021.txt
"""

import argparse
from pathlib import Path


def merge_scores(input_files: list[Path], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    with output_file.open("w", encoding="utf-8") as fout:
        for file_path in input_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Missing input score file: {file_path}")

            line_count = 0
            with file_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        fout.write(line + "\n")
                        line_count += 1

            total_lines += line_count
            print(f"{file_path.name}: {line_count} lines")

    print(f"Saved merged score file to: {output_file}")
    print(f"Total lines: {total_lines}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
        help="Input score files",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Merged output score file",
    )

    args = parser.parse_args()
    merge_scores(args.inputs, args.output)


if __name__ == "__main__":
    main()
