""" Base model class.
"""

from abc import abstractmethod
from typing import Any, Dict
import torch
import pytorch_lightning as pl


class Model(pl.LightningModule):
    """Base model class."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add command line arguments."""
        group = parser.add_argument_group("Model")
        group.add_argument("--max_seq_len", default=512, type=int,
                           help="The maximum length of the sequence. Default: 512.")
        group.add_argument("--use_fast_tokenizer", default='false', type=str,
                           help="Whether or not to try to load the fast version of the tokenizer. Default: false.")
        group.add_argument("--special_tokens", type=str,
                           help="The config file of special tokens.")
        group.add_argument("--optimizer", default="AdamW", type=str, choices=["Adam", "AdamW", "Adafactor"],
                           help="The optimizer for training. Choices: ['Adam', 'AdamW', 'Adafactor']. "
                                "Default: 'AdamW'.")
        group.add_argument("--scheduler", default="linear", type=str, choices=["constant", "linear", "noam"],
                           help="The learning rate scheduler for training. Choices: ['constant', 'linear', 'noam']. "
                                "Default: 'linear'.")
        group.add_argument("-lr", "--learning_rate", default=1e-5, type=float,
                           help="The peak learning rate for optimizer. Default: 1e-5.")
        group.add_argument("--warmup_steps_ratio", default=0.1, type=float,
                           help="The ratio of warmup steps of total steps. Default: 0.1.")
        group.add_argument("--weight_decay", default=0.0, type=float, help="The weight decay for optimizer.")

        return group

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.max_seq_len = args.max_seq_len
        self.use_fast_tokenizer = args.use_fast_tokenizer in ('true', '1')
        self.special_tokens = args.special_tokens
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.lr = args.learning_rate
        self.warmup_steps_ratio = args.warmup_steps_ratio
        self.weight_decay = args.weight_decay

    @abstractmethod
    def setup_model(self):
        """Set up model structure.

        Raises:
            NotImplementedError: This abstract method needs to be implemented in subclasses.
        """
        raise NotImplementedError

    def forward(self, inputs):
        """Model forward.

        Args:
            inputs (dict): A dict mapping keys to corresponding input data.

        Returns:
            dict: A dict mapping keys to corresponding output data.
        """
        outputs = self.model(
            inputs["input_ids"],
            token_type_ids=inputs.get("token_type_ids"),
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
            return_dict=True
        )
        return outputs

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training loop.

        Args:
            batch (list): The output of DataLoader.
            batch_idx (int): Integer displaying index of this batch.

        Raises:
            NotImplementedError: This abstract method needs to be implemented in subclasses.

        Returns:
            dict: A training outputs dictionary which must include the key 'loss'.
        """
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation loop.

        Args:
            batch (list): The output of DataLoader.
            batch_idx (int): Integer displaying index of this batch.

        Raises:
            NotImplementedError: This abstract method needs to be implemented in subclasses.

        Returns:
            dict: A validation outputs dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step.

        Args:
            batch (list): The output of DataLoader.
            batch_idx (int): Integer displaying index of this batch.

        Raises:
            NotImplementedError: This abstract method needs to be implemented in subclasses.

        Returns:
            dict: A test outputs dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction loop.

        Args:
            batch (list): The output of DataLoader.
            batch_idx (int): Integer displaying index of this batch.

        Raises:
            NotImplementedError: This abstract method needs to be implemented in subclasses.

        Returns:
            dict: A prediction outputs dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self):
        """Configure optimizers and learning-rate schedulers.

        Raises:
            NotImplementedError: This abstract method needs to be implemented in subclasses.

        Returns:
            dict: A optimizer configuration dictionary which must include the key 'optimizer'.
        """
        raise NotImplementedError

    @abstractmethod
    def create_inputs(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Create model inputs dictionary from dataloader batch list.

        Args:
            batch (list): The output of DataLoader.

        Raises:
            NotImplementedError: This abstract method needs to be implemented in subclasses.

        Returns:
            dict: A model inputs dictionary.
        """
        raise NotImplementedError

    def interact(self, dm):
        """ Coupling Model with DataModule.
        """
        dm.register_fn(self.create_inputs)
        if self.special_tokens:
            self.model.resize_token_embeddings(len(dm.tokenizer))
