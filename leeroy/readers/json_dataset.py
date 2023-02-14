""" Json dataset.
"""

import json

import torch


class JsonDataset(torch.utils.data.Dataset):
    """Json dataset.
    """

    def __init__(
        self,
        data_file,
    ):
        super().__init__()
        self.data_file = data_file
        self._data = [json.loads(ln) for ln in open(self.data_file, encoding='utf-8')]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
