"""Auto DataModule.
Create DataModule by file suffix.
"""
import os
from functools import partial
import torch
from typing import Callable, Optional
from transformers import AutoTokenizer

from .csv_dataset import CsvDataset
from .json_dataset import JsonDataset
from .datamodule_base import DataModule


class AutoDataModule(DataModule):
    """Auto DataModule.
    """

    def __init__(self, args):
        super().__init__(args)

        self.collate_fn = None
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=args.use_fast_tokenizer)
        if args.special_tokens:
            with open(args.special_tokens, "r") as fin:
                special_tokens = [w.strip() for w in fin]
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def register_fn(self, cfn: Callable):
        self.collate_fn = partial(cfn, tokenizer=self.tokenizer)

    def prepare_data(self):
        if self.train_file and not os.path.isfile(self.train_file):
            raise ValueError(f"`{self.train_file}` does not exist.")
        if self.valid_file and not os.path.isfile(self.valid_file):
            raise ValueError(f"`{self.valid_file}` does not exist.")
        if self.test_file and not os.path.isfile(self.test_file):
            raise ValueError(f"`{self.test_file}` does not exist.")
        if self.predict_file and not os.path.isfile(self.predict_file):
            raise ValueError(f"`{self.predict_file}` does not exist.")

    def setup(self, stage: Optional[str] = 'fit'):
        """Set up dataset, which will be called on every process in DDP.

        Args:
            stage (str): A string flag of data stage: fit/test/predict.
        """
        if stage == "fit":
            self.train_dataset = AutoDataModule._auto_create_dataset(self.train_file)
            self.valid_dataset = AutoDataModule._auto_create_dataset(self.valid_file)
        elif stage == "test":
            self.test_dataset = AutoDataModule._auto_create_dataset(self.test_file)
        elif stage == "predict":
            self.predict_dataset = AutoDataModule._auto_create_dataset(self.predict_file)

    def train_dataloader(self):
        """Set up training dataloader.

        Returns:
            torch.utils.data.DataLoader: Training dataloader.
        """
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        """Set up validation dataloader.

        Returns:
            torch.utils.data.DataLoader: Validation dataloader.
        """
        return torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        """Set up test dataloader.

        Returns:
            torch.utils.data.DataLoader: Test dataloader.
        """
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        """Set up predict dataloader.

        Returns:
            torch.utils.data.DataLoader: Predict dataloader.
        """
        return torch.utils.data.DataLoader(
            self.predict_dataset, batch_size=self.batch_size, num_workers=1, collate_fn=self.collate_fn)

    @staticmethod
    def _auto_create_dataset(file):
        if file.endswith('.csv'):
            return CsvDataset(file)
        elif file.endswith('.json'):
            return JsonDataset(file)
        else:
            raise ValueError('Unsupported file type. Only supports [csv, json]')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    AutoDataModule.add_cmdline_args(parser)
    args = parser.parse_args()
    args.predict_file = '/home/huangliankai/code/Leeroy/examples/span_extraction/test.json'
    args.batch_size = 1
    print(args)
    dm = AutoDataModule(args)
    dm.setup('predict')
    for batch in dm.predict_dataloader():
        print(batch)
        break
