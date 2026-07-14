"""Data helpers for the plain (non-torchdata) EMG pipeline.

One recording == one CSV file. Each row is ``ch0[, ch1, ...], label`` where the first
``channel`` columns are the EMG sample(s) and the last column is an integer class id.
"""
import os
from typing import List, Tuple

import numpy as np


def list_csv_files(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    files = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"no .csv files under {data_dir}")
    return files


def read_recording(path: str, channel: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read one CSV recording.

    Returns (data[N, channel] float32, labels[N] int64). Assumes a header row.
    The first ``channel`` columns are features; the LAST column is the label.
    """
    # skip header, be robust to blank trailing lines
    rows = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64)
    if rows.ndim == 1:  # single row
        rows = rows[None, :]
    # empty / header-only / too-few-columns file -> return empty (Dataset skips short recordings)
    if rows.size == 0 or rows.shape[1] < channel + 1:
        return np.empty((0, channel), np.float32), np.empty((0,), np.int64)
    rows = rows[~np.isnan(rows).any(axis=1)]  # drop malformed/blank rows
    data = rows[:, :channel].astype(np.float32)
    labels = rows[:, -1].astype(np.int64)
    return data, labels


def compute_norm_stats(files, channel: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over the given training files (for z-score standardization).

    ``files`` may be a directory path or an explicit list of CSV paths.
    """
    if isinstance(files, str):
        files = list_csv_files(files)
    chunks = [x for x in (read_recording(p, channel)[0] for p in files) if len(x)]
    allx = np.concatenate(chunks, axis=0)
    mean = allx.mean(axis=0)
    std = allx.std(axis=0)
    std[std < 1e-8] = 1.0  # guard against constant channels
    return mean.astype(np.float32), std.astype(np.float32)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-frequency class weights, normalized to mean 1.0. Handles missing classes."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero for absent classes
    inv = counts.sum() / (num_classes * counts)
    inv = inv / inv.mean()
    return inv.astype(np.float32)
