"""Base datamodule.
"""

from abc import abstractmethod

import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    """Base data module class."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add command line arguments."""
        group = parser.add_argument_group("DataModule")
        group.add_argument("--train_file", default="", type=str,
                           help="The training dataset file. Default: ''.")
        group.add_argument("--valid_file", default="", type=str,
                           help="The validation dataset file. Default: ''.")
        group.add_argument("--test_file", default="", type=str,
                           help="The test dataset file. Default: ''.")
        group.add_argument("--predict_file", default="", type=str,
                           help="The prediction dataset file. Default: ''.")
        group.add_argument("--batch_size", default=32, type=int,
                           help="The size of batches. Default: 32. ")
        group.add_argument("--num_workers", default=8, type=int,
                           help="The number of workers used in data loader. Default: 8.")
        group.add_argument("--tokenizer_name_or_path", type=str,
                           help="Use `model_name_or_path` by default")
        return group

    def __init__(self, args):
        super().__init__()
        self.train_file = args.train_file
        self.valid_file = args.valid_file
        self.test_file = args.test_file
        self.predict_file = args.predict_file
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        if not args.tokenizer_name_or_path:
            args.tokenizer_name_or_path = args.model_name_or_path

    def prepare_data(self):
        """Prepare dataset, which will only be called on GPU:0 in distributed.

        Raises:
            NotImplementedError: This abstract method needs to be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def setup(self, stage=None):
        """Set up dataset, which will be called on every process in DDP.

        Args:
            stage (str): A string flag of data stage: train/valid/test.

        Raises:
            NotImplementedError: This abstract method needs to be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def train_dataloader(self):
        """Set up training dataloader.

        Raises:
            NotImplementedError: This abstract method needs to be implemented in subclasses.

        Returns:
            torch.utils.data.DataLoader: Training dataloader.
        """
        raise NotImplementedError

    @abstractmethod
    def val_dataloader(self):
        """Set up validation dataloader.

        Raises:
            NotImplementedError: This abstract method needs to be implemented in subclasses.

        Returns:
            torch.utils.data.DataLoader: Validation dataloader.
        """
        raise NotImplementedError

    @abstractmethod
    def test_dataloader(self):
        """Set up test dataloader.

        Raises:
            NotImplementedError: This abstract method needs to be implemented in subclasses.

        Returns:
            torch.utils.data.DataLoader: Test dataloader.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_dataloader(self):
        """Set up prediction dataloader.

        Raises:
            NotImplementedError: This abstract method needs to be implemented in subclasses.

        Returns:
            torch.utils.data.DataLoader: Prediction dataloader.
        """
        raise NotImplementedError
