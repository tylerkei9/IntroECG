from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EchoNextWaveformDataset(Dataset):
    """Memory-efficient dataset for EchoNext waveform arrays.

    The waveform file is opened with NumPy mmap so samples are read from disk
    on demand instead of materializing the full split in RAM.
    """

    def __init__(self, waveform_path, labels):
        self.waveform_path = Path(waveform_path)
        self.waveforms = np.load(self.waveform_path, mmap_mode="r")
        self.labels = np.asarray(labels, dtype=np.float32)

        if len(self.waveforms) != len(self.labels):
            raise ValueError(
                f"Waveform rows ({len(self.waveforms)}) do not match labels ({len(self.labels)})"
            )

        expected_shape = (1, 2500, 12)
        if self.waveforms.shape[1:] != expected_shape:
            raise ValueError(
                f"Expected waveform shape (N, 1, 2500, 12), got {self.waveforms.shape}"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        waveform = torch.tensor(self.waveforms[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return waveform, label
