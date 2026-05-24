"""Split data/laws_de.csv into ~30 MB CSV segments, preserving the header in each."""
import csv
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "data" / "laws_de.csv"
OUT_DIR = SRC.parent / "laws_de_segments"
TARGET_BYTES = 30 * 1024 * 1024

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for old in OUT_DIR.glob("laws_de_part*.csv"):
        old.unlink()

    with SRC.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        part_idx = 1
        out_path = OUT_DIR / f"laws_de_part{part_idx:02d}.csv"
        out_f = out_path.open("w", encoding="utf-8", newline="")
        writer = csv.writer(out_f)
        writer.writerow(header)
        rows_in_part = 0
        total_rows = 0

        def rotate() -> tuple:
            nonlocal part_idx, out_f, writer, rows_in_part
            size = out_f.tell()
            out_f.close()
            print(f"  -> {out_path.name}: {rows_in_part} rows, {size / (1024 * 1024):.2f} MB")
            part_idx += 1
            new_path = OUT_DIR / f"laws_de_part{part_idx:02d}.csv"
            new_f = new_path.open("w", encoding="utf-8", newline="")
            new_writer = csv.writer(new_f)
            new_writer.writerow(header)
            rows_in_part = 0
            return new_path, new_f, new_writer

        for row in reader:
            writer.writerow(row)
            rows_in_part += 1
            total_rows += 1
            if out_f.tell() >= TARGET_BYTES:
                out_path, out_f, writer = rotate()

        size = out_f.tell()
        out_f.close()
        print(f"  -> {out_path.name}: {rows_in_part} rows, {size / (1024 * 1024):.2f} MB")

    print(f"Done. {total_rows} data rows split into {part_idx} parts in {OUT_DIR}")


if __name__ == "__main__":
    main()
