"""Data sanity check: mean fraction of 'rest' (class 0) labels across recordings.

A high value signals class imbalance (train.py handles it via class-weighted loss and
reports macro-F1 / confusion, not just top-line accuracy).
"""
import argparse

import numpy as np

from datasets.utils import list_csv_files, read_recording


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="emg_data/train")
    p.add_argument("--channel", type=int, default=1)
    p.add_argument("--num-classes", type=int, default=4)
    args = p.parse_args()

    fractions, class_counts = [], np.zeros(args.num_classes, dtype=np.int64)
    files = list_csv_files(args.data_dir)
    for path in files:
        _, labels = read_recording(path, args.channel)
        fractions.append(np.mean(labels == 0))
        class_counts += np.bincount(labels, minlength=args.num_classes)

    print(f"files: {len(files)}")
    print(f"mean fraction of class 0 (rest) per file: {np.mean(fractions):.4f}")
    total = class_counts.sum()
    print("overall class distribution:")
    for c, n in enumerate(class_counts):
        print(f"  class {c}: {n:>8d}  ({100 * n / max(total, 1):.2f}%)")


if __name__ == "__main__":
    main()
