"""Multilabel classification model.
"""
from typing import Any, Dict, List

import torch
from torchmetrics import Metric, Accuracy, Precision, Recall, F1Score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .multi_classification import MultiClassificationModel


class MultiLabelClassificationModel(MultiClassificationModel):
    """Model for the multilabel classification task.
    """
    @classmethod
    def add_cmdline_args(cls, parser):
        """Add command line arguments."""
        group = super().add_cmdline_args(parser)
        group.add_argument("--threshold", type=float, default=0.5,
            help="Threshold for transforming probability to binary (0,1) predictions. Default: 0.5.")
        return group

    def __init__(self, args) -> None:
        self.threshold = args.threshold
        super().__init__(args)
        self.sigmoid = torch.nn.Sigmoid()

    def setup_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, num_labels=self.num_classes, problem_type="multi_label_classification")

    def _configure_metrics(self) -> Dict[str, Metric]:
        metrics = {}
        for stage in ('val', 'test'):
            metrics[f'{stage}_precision'] = Precision(num_classes=self.num_classes, average="macro", threshold=self.threshold)
            metrics[f'{stage}_recall'] = Recall(num_classes=self.num_classes, average="macro", threshold=self.threshold)
            metrics[f'{stage}_f1'] = F1Score(num_classes=self.num_classes, average="macro", threshold=self.threshold)
            metrics[f'{stage}_accuracy'] = Accuracy()
        return metrics

    def _common_step(self, stage: str, batch: Any) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch)
        loss = outputs.loss
        logits = outputs.logits
        preds = self.sigmoid(logits)
        if batch["labels"] is not None:
            batch_size=len(batch["labels"])
            stage_metrics = self._update_metrics(preds, batch["labels"].long(), stage=stage)
            self.log_dict(stage_metrics, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return {'loss': loss}

    def predict_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch)
        logits = outputs.logits
        probs = self.sigmoid(logits)
        preds = torch.where(probs >= self.threshold, 1, 0)
        probs = [pb.tolist() for pb in probs]
        preds = [torch.nonzero(pd, as_tuple=True)[0].tolist() for pd in preds]
        return {
            'probs': probs,
            'preds': preds,
        }

    def create_inputs(self, batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
        """ Write your custom inputs creating function.
        """
        def _create_labels(batch_label):
            index = [[int(y) for y in ys.split()] for ys in batch_label]
            labels = torch.zeros((len(batch_label), self.num_classes), dtype=torch.float)
            for i in range(len(index)):
                for j in index[i]:
                    labels[i][j] = 1
            return labels

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
        if hasattr(batch, 'label'):
            inputs["labels"] = None
        else:
            inputs["labels"] = _create_labels([x["label"] for x in batch])

        return inputs