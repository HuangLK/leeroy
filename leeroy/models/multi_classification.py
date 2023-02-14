"""Multi-class classification model.
"""

from typing import Any, Dict, List

import torch
from torchmetrics import Metric, Accuracy, Precision, Recall, F1Score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .model_base import Model
from .common import get_optimizer_scheduler


class MultiClassificationModel(Model):
    """Model for the multi-class classification task.
    """
    task = 'mulclass_clf'

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
        self.setup_metrics()

    def setup_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, num_labels=self.num_classes)

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return {'loss': loss}

    def _common_step(self, stage: str, batch: Any) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        if batch["labels"] is not None:
            batch_size=len(batch["labels"])
            stage_metrics = self._update_metrics(preds, batch["labels"], stage=stage)
            self.log_dict(stage_metrics, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return {'loss': loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._common_step("test", batch)

    def predict_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        inputs = self.create_inputs(batch, stage='predict')
        outputs = self.forward(inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        return {'preds': preds}

    def _configure_metrics(self) -> Dict[str, Metric]:
        metrics = {}
        # 类别个位为2时，则会分别计算两个类别的准召
        for stage in ('val', 'test'):
            metrics[f'{stage}_precision'] = Precision(num_classes=self.num_classes, average="macro")
            metrics[f'{stage}_recall'] = Recall(num_classes=self.num_classes, average="macro")
            metrics[f'{stage}_f1'] = F1Score(num_classes=self.num_classes, average="macro")
            metrics[f'{stage}_accuracy'] = Accuracy()
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
        assert 'text_1' in batch[0], 'data fields'
        # Either encode single sentence or sentence pairs
        if 'text_2' in batch[0]:
            texts_or_text_pairs = [(x['text_1'], x['text_2']) for x in batch]
        else:
            texts_or_text_pairs = [x['text_1'] for x in batch]

        inputs = tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
        )
        if 'label' not in batch[0]:
            inputs["labels"] = None
        else:
            inputs["labels"] = torch.tensor([int(x["label"]) for x in batch], dtype=torch.long)

        return inputs