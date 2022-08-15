"""Multi-class classification model.
"""

from typing import Any, Dict

import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.model_base import Model
from models.common import get_optimizer_scheduler


class MultiClassClassificationModel(Model):
    """Model for the multi-class classification task.
    """
    @classmethod
    def add_cmdline_args(cls, parser):
        """Add command line arguments."""
        group = super().add_cmdline_args(parser)
        group.add_argument("--model_name_or_path", default="hfl/chinese-macbert-base", type=str,
                           help="BERT-like model in huggingface.co or a path to model weights. "
                                "Default: 'hfl/chinese-macbert-base'.")
        group.add_argument("--num_classes", type=int, default=-1, help="Number of classification classes ")
        return group

    def __init__(self, args) -> None:
        super().__init__(args)
        self.model_name_or_path = args.model_name_or_path
        self.num_classes = args.num_classes
        assert self.num_classes > 1, f'argment error: num_classes must be greater than 1.'
        self.setup_model()
        self.configure_metrics()

    def setup_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=self.use_fast_tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, num_labels=self.num_classes)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs = self.create_inputs(batch)
        outputs = self.forward(inputs)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def _common_step(self, stage: str, batch: Any) -> torch.Tensor:
        inputs = self.create_inputs(batch, stage=stage)
        outputs = self.forward(inputs)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        if inputs["labels"] is not None:
            batch_size=len(inputs["labels"])
            metric_dict = self._compute_metrics(preds, inputs["labels"], stage=stage)
            self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self._common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self._common_step("test", batch)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        inputs = self.create_inputs(batch, stage='predict')
        outputs = self.forward(inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_metrics(self) -> None:
        # 类别个位为2时，则会分别计算两个类别的准召
        prec = Precision(num_classes=self.num_classes, average="macro")
        recall = Recall(num_classes=self.num_classes, average="macro")
        f1 = F1Score(num_classes=self.num_classes, average="macro")
        acc = Accuracy()
        self.metrics = {"precision": prec, "recall": recall, "f1": f1, "accuracy": acc}

    def _compute_metrics(self, preds, labels, stage="val") -> Dict[str, torch.Tensor]:
        return {f"{stage}_{k}": metric(preds.cpu(), labels.cpu()) for k, metric in self.metrics.items()}

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

    def create_inputs(self, batch: Any, stage: str = 'fit') -> Dict[str, torch.Tensor]:
        """ Write your custom inputs creating function.
        """
        assert 'text_1' in batch, 'data fields'
        # Either encode single sentence or sentence pairs
        if 'text_2' in batch:
            texts_or_text_pairs = list(zip(batch['text_1'], batch['text_2']))
        else:
            texts_or_text_pairs = batch['text_1']

        inputs = self._tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
        )
        if stage == 'predict':
            inputs["labels"] = None
        else:
            inputs["labels"] = torch.tensor([int(y) for y in batch["label"]], dtype=torch.long)

        if torch.cuda.is_available():
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.cuda()

        return inputs
