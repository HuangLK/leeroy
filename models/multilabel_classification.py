"""Multilabel classification model.
"""
from typing import Any, Dict

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
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=self.use_fast_tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, num_labels=self.num_classes, problem_type="multi_label_classification")
        if self.special_tokens:
            with open(self.special_tokens, "r") as fin:
                special_tokens = [w.strip() for w in fin]
            self._tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.model.resize_token_embeddings(len(self._tokenizer))

    def _configure_metrics(self) -> Dict[str, Metric]:
        metrics = {}
        for stage in ('val', 'test'):
            metrics[f'{stage}_precision'] = Precision(num_classes=self.num_classes, average="macro", threshold=self.threshold)
            metrics[f'{stage}_recall'] = Recall(num_classes=self.num_classes, average="macro", threshold=self.threshold)
            metrics[f'{stage}_f1'] = F1Score(num_classes=self.num_classes, average="macro", threshold=self.threshold)
            metrics[f'{stage}_accuracy'] = Accuracy()
        return metrics

    def _common_step(self, stage: str, batch: Any) -> Dict[str, torch.Tensor]:
        inputs = self.create_inputs(batch, stage=stage)
        outputs = self.forward(inputs)
        loss = outputs.loss
        logits = outputs.logits
        preds = self.sigmoid(logits)
        if inputs["labels"] is not None:
            batch_size=len(inputs["labels"])
            stage_metrics = self._update_metrics(preds, inputs["labels"].long(), stage=stage)
            self.log_dict(stage_metrics, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return {'loss': loss}

    def predict_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        inputs = self.create_inputs(batch, stage='predict')
        outputs = self.forward(inputs)
        logits = outputs.logits
        probs = self.sigmoid(logits)
        preds = torch.where(probs >= self.threshold, 1, 0)
        probs = [pb.tolist() for pb in probs]
        preds = [torch.nonzero(pd, as_tuple=True)[0].tolist() for pd in preds]
        return {
            'probs': probs,
            'preds': preds,
        }

    def create_inputs(self, batch: Any, stage: str = 'fit') -> Dict[str, torch.Tensor]:
        """ Write your custom inputs creating function.
        """
        def _create_labels(batch_label):
            index = [[int(y) for y in ys.split()] for ys in batch_label]
            labels = torch.zeros((len(batch_label), self.num_classes), dtype=torch.float)
            for i in range(len(index)):
                for j in index[i]:
                    labels[i][j] = 1
            return labels

        assert 'text_1' in batch, 'data fields'
        # Either encode single sentence or sentence pairs
        if 'text_2' in batch:
            texts_or_text_pairs = list(zip(batch['text_1'], batch['text_2']))
        else:
            texts_or_text_pairs = batch['text_1']

        inputs = self._tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
        )
        if stage == 'predict':
            inputs["labels"] = None
        else:
            inputs["labels"] = _create_labels(batch["label"])

        if torch.cuda.is_available():
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.cuda()

        return inputs
