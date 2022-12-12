"""T5 model.
"""

from typing import Any, Dict, List

import torch
from torchmetrics import Metric, Accuracy, Precision, Recall, F1Score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from models.model_base import Model
from models.common import get_optimizer_scheduler


class T5(Model):
    """T5 model.
    """
    @classmethod
    def add_cmdline_args(cls, parser):
        """Add command line arguments."""
        group = super().add_cmdline_args(parser)
        group.add_argument("--model_name_or_path", default="Langboat/mengzi-t5-base-mt", type=str,
                           help="BERT-like model in huggingface.co or a path to model weights. "
                                "Default: 'Langboat/mengzi-t5-base-mt'.")
        group.add_argument("--source_max_seq_len", type=int, default=512)
        group.add_argument("--target_max_seq_len", type=int, default=128)

        return group

    def __init__(self, args) -> None:
        super().__init__(args)
        self.model_name_or_path = args.model_name_or_path
        self.source_max_seq_len = args.source_max_seq_len
        self.target_max_seq_len = args.target_max_seq_len
        self.setup_model()
        self.setup_metrics()

    def setup_model(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)

    def forward(self, inputs):
        """Model forward. See more detail: https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/t5#training

        Args:
            inputs (dict): A dict mapping keys to corresponding input data.

        Returns:
            dict: A dict mapping keys to corresponding output data.
        """
        outputs = self.model(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
            return_dict=True
        )
        return outputs

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return {'loss': loss}

    def _common_step(self, stage: str, batch: Any) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch)
        loss = outputs.loss
        if batch["labels"] is not None:
            batch_size=len(batch["labels"])
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return {'loss': loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._common_step("test", batch)

    def predict_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        inputs = self.create_inputs(batch, stage='predict')
        outputs = self.forward(inputs)
        # TODO: predict step
        preds = None
        return {'preds': preds}

    def _configure_metrics(self) -> Dict[str, Metric]:
        metrics = {}
        return metrics

    def setup_metrics(self) -> None:
        self.metrics = self._configure_metrics()
        for k, v in self.metrics.items():
            setattr(self, k, v)

    def _update_metrics(self, preds, labels, stage="val") -> Dict[str, Metric]:
        _ = [metric.update(preds, labels) for k, metric in self.metrics.items() if k.startswith(stage)]
        return {k: metric for k, metric in self.metrics.items() if k.startswith(stage)}

    def configure_optimizers(self) -> Dict:
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(self.warmup_steps_ratio * num_training_steps)
        optimizers_schedulers = get_optimizer_scheduler(
            self.model,
            self.optimizer,
            self.scheduler,
            self.lr,
            self.weight_decay,
            num_warmup_steps,
            num_training_steps,
        )
        return optimizers_schedulers

    def create_inputs(self, batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
        """ Write your custom inputs creating function.
        """
        assert 'source_text' in batch[0], 'data fields error'

        inputs = tokenizer.batch_encode_plus(
            [x['source_text'] for x in batch],
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.source_max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
        )
        if 'target_text' not in batch[0]:
            inputs["target_text"] = None
        else:
            target_token = tokenizer.batch_encode_plus(
                [x['target_text'] for x in batch],
                add_special_tokens=True,
                return_attention_mask=True,
                max_length=self.target_max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors='pt',
            )
            labels = target_token['input_ids']
            # replace padding token id's of the labels by -100 so it's ignored by the loss
            labels[labels == tokenizer.pad_token_id] = -100
            inputs['labels']  = labels

        return inputs
