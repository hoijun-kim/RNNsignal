"""Map-style EMG windowing dataset (replaces the old torchdata datapipe).

Why map-style instead of the previous torchdata IterDataPipe:
  * windows are built PER FILE, so a window never straddles two recordings
    (the old rolling pipe concatenated all files into one stream and mixed them);
  * windows are materialized once, not re-parsed/re-windowed every epoch;
  * __len__ works, so training needs no ``len(list(enumerate(loader)))`` full pass,
    and shuffling / a real validation loop are straightforward.

Targets are integer class ids (for nn.CrossEntropyLoss), not one-hot.
"""
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.utils import list_csv_files, read_recording


class EMGWindowDataset(Dataset):
    def __init__(self, data_dir: Optional[str] = None, window_size: int = 64, channel: int = 1,
                 num_classes: int = 4, step: int = 1,
                 mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None,
                 files: Optional[list] = None):
        self.window_size = window_size
        self.channel = channel
        self.num_classes = num_classes
        self.step = step
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

        if files is None:
            files = list_csv_files(data_dir)
        data_windows, label_windows = [], []
        for path in files:
            data, labels = read_recording(path, channel)  # (N, C), (N,)
            n = len(labels)
            if n < window_size:
                continue  # recording too short for even one window
            for i in range(0, n - window_size + 1, step):
                data_windows.append(data[i:i + window_size])
                label_windows.append(labels[i:i + window_size])

        if not data_windows:
            raise ValueError(f"no windows built from {data_dir} (window_size={window_size} too large?)")

        self.data = np.stack(data_windows).astype(np.float32)     # (M, W, C)
        self.labels = np.stack(label_windows).astype(np.int64)    # (M, W)

        bad = (self.labels < 0) | (self.labels >= num_classes)
        if bad.any():
            raise ValueError(f"labels outside [0, {num_classes - 1}] found in {data_dir}")

        if self.mean is not None and self.std is not None:
            self.data = (self.data - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "data": torch.from_numpy(self.data[idx]),      # (W, C) float32
            "label": torch.from_numpy(self.labels[idx]),   # (W,) int64
        }


# Backwards-compatible constructor name.
def emg_dataset(data_dir: str = "./emg_data/train", window_size: int = 64, channel: int = 1,
                num_classes: int = 4, step: int = 1, mean=None, std=None) -> EMGWindowDataset:
    return EMGWindowDataset(data_dir, window_size, channel, num_classes, step, mean, std)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    ds = emg_dataset("./emg_data/train", window_size=64, channel=1, num_classes=4, step=1)
    print("windows:", len(ds))
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    b = next(iter(dl))
    print("data", tuple(b["data"].shape), b["data"].dtype, "| label", tuple(b["label"].shape), b["label"].dtype)
