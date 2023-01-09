"""T5 model.
"""

from typing import Any, Dict, List

import torch
from torchmetrics import Metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from models.model_base import Model
from models.common import get_optimizer_scheduler


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder
        self.main_input_name = encoder.main_input_name

    def forward(self, input_ids = None, attention_mask = None, **kwargs):
        #assert attention_mask is None, f'no need to provide `attention_mask` of encoder'
        bs = input_ids.size(0)
        seq_len = input_ids.size(1) // self.n_context
        input_ids = input_ids.view(bs * self.n_context, -1)
        if attention_mask is not None:
            attention_mask = attention_mask.view(bs * self.n_context, -1)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        if not kwargs.get('return_dict', False):
            outputs = (outputs[0].view(bs, self.n_context * seq_len, -1), ) + outputs[1:]
        else:
            outputs.last_hidden_state = outputs.last_hidden_state.view(bs, self.n_context * seq_len, -1)
        return outputs

    def set_input_embeddings(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


class FID(Model):
    """Fusion-in-Decoder model.
    Acknowledge: https://github.com/facebookresearch/FiD
    """
    task = 'fid'

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add command line arguments."""
        group = super().add_cmdline_args(parser)
        group.add_argument("--model_name_or_path", default="Langboat/mengzi-t5-base-mt", type=str,
                           help="T5 model in huggingface.co or a path to model weights. "
                                "Default: 'Langboat/mengzi-t5-base-mt'.")
        group.add_argument("--source_max_seq_len", type=int, default=512)
        group.add_argument("--target_max_seq_len", type=int, default=128)
        group.add_argument("--n_context", type=int, default=10)

        return group

    def __init__(self, args) -> None:
        super().__init__(args)
        self.model_name_or_path = args.model_name_or_path
        self.source_max_seq_len = args.source_max_seq_len
        self.target_max_seq_len = args.target_max_seq_len
        self.n_context = args.n_context
        self.setup_model()
        self.setup_metrics()

    def setup_model(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)
        self.model.encoder = EncoderWrapper(self.model.encoder)
        self.model.encoder.n_context = self.n_context

    def generate(self, **kwargs):
        if kwargs["input_ids"].dim() == 3:
            bs, n_context, seq_len = kwargs["input_ids"].shape
            kwargs["input_ids"] = kwargs["input_ids"].view(bs, n_context * seq_len)
            if 'attention_mask' in kwargs:
                kwargs["attention_mask"] = kwargs["attention_mask"].view(bs, n_context * seq_len)

        return self.model.generate(**kwargs)

    def forward(self, inputs):
        """Model forward.

        Args:
            inputs (dict): A dict mapping keys to corresponding input data.

        Returns:
            dict: A dict mapping keys to corresponding output data.
        """
        # We need to resize as B x (N * L) instead of (B * N) x L here.
        # because the T5 forward method uses the input tensors to *infer* dimensions used in the decoder.
        # EncoderWrapper resizes the inputs as (B * N) x L.
        if inputs["input_ids"].dim() == 3:
            bs, n_context, seq_len = inputs["input_ids"].shape
            inputs["input_ids"] = inputs["input_ids"].view(bs, n_context * seq_len)
            if 'attention_mask' in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].view(bs, n_context * seq_len)

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
        assert 'context' in batch[0], 'data fields error'

        for x in batch:
            x['context'].extend(['无'] * (self.n_context - len(x['context'])))

        PASSAGE_FORMAT = '问题：{} 知识：{}'
        bs = len(batch)
        flatten_passages = [PASSAGE_FORMAT.format(x['source_text'], p) for x in batch for p in x['context'][: self.n_context]]
        flatten_inputs = tokenizer.batch_encode_plus(
            flatten_passages,
            add_special_tokens=True,
            return_attention_mask=False,
            max_length=self.source_max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
        )

        inputs = {}
        for k, v in flatten_inputs.items():
            inputs[k] = v.unflatten(0, (bs, self.n_context))

        if 'target_text' not in batch[0]:
            inputs["target_text"] = None
        else:
            target_token = tokenizer.batch_encode_plus(
                [x['target_text'] for x in batch],
                add_special_tokens=True,
                return_attention_mask=False,
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
