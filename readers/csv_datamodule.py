"""CSV DataModule.
"""
import os
import torch
import pytorch_lightning as pl

from readers.csv_dataset import CsvDataset
from readers.datamodule_base import DataModule


class CSVDataModule(DataModule):
    """CSV DataModule.
    """

    def __init__(self, args):
        super().__init__(args)

    def prepare_data(self):
        if self.train_file and not os.path.isfile(self.train_file):
            raise ValueError(f"`{self.train_file}` does not exist.")
        if self.valid_file and not os.path.isfile(self.valid_file):
            raise ValueError(f"`{self.valid_file}` does not exist.")
        if self.test_file and not os.path.isfile(self.test_file):
            raise ValueError(f"`{self.test_file}` does not exist.")
        if self.predict_file and not os.path.isfile(self.predict_file):
            raise ValueError(f"`{self.predict_file}` does not exist.")

    def setup(self, stage='fit'):
        """Set up dataset, which will be called on every process in DDP.

        Args:
            stage (str): A string flag of data stage: fit/test/predict.
        """
        if stage == "fit":
            self.train_dataset = CsvDataset(self.train_file)
            self.valid_dataset = CsvDataset(self.valid_file)
        elif stage == "test":
            self.test_dataset = CsvDataset(self.test_file)
        elif stage in ("predict", "infer"):
            self.predict_dataset = CsvDataset(self.predict_file)

    def train_dataloader(self):
        """Set up training dataloader.

        Returns:
            torch.utils.data.DataLoader: Training dataloader.
        """
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        """Set up validation dataloader.

        Returns:
            torch.utils.data.DataLoader: Validation dataloader.
        """
        return torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        """Set up test dataloader.

        Returns:
            torch.utils.data.DataLoader: Test dataloader.
        """
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        """Set up predict dataloader.

        Returns:
            torch.utils.data.DataLoader: Predict dataloader.
        """
        return torch.utils.data.DataLoader(
            self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    CSVDataModule.add_cmdline_args(parser)
    args = parser.parse_args()
    args.train_file = '/home/huangliankai/code/kg_base/factual_consistency/train.csv'
    args.valid_file = '/home/huangliankai/code/kg_base/factual_consistency/val.csv'
    args.batch_size = 1
    print(args)
    dm = CSVDataModule(args)
    dm.setup('fit')
    for batch in dm.train_dataloader():
        print(batch)
        break