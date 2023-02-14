""" CSV dataset.
"""

import torch

from ..utils.csv_utils import load_csv


class CsvDataset(torch.utils.data.Dataset):
    """CSV dataset.
    """

    def __init__(
        self,
        data_file,
    ):
        super().__init__()
        self.data_file = data_file
        self._data = list(load_csv(self.data_file))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
