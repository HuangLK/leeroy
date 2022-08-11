""" CSV dataset.
"""

import csv

import torch
from transformers import AutoTokenizer


class CsvDataset(torch.utils.data.Dataset):
    """CSV dataset.
    """

    def __init__(
        self,
        data_file,
    ):
        super().__init__()
        self.data_file = data_file
        self._data = list(self._iter_data())

    def _iter_data(self):
        with open(self.data_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            headers = next(reader)
            for row in reader:
                yield { k: v for k, v in zip(headers, row) }

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
