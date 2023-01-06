"""Span-based model.
"""
from typing import Any, Dict, Optional, Union, Tuple

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from .model_base import Model
from utils import span_utils as sputils


class BertForSpan(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        self.linear_start = nn.Linear(config.hidden_size, 1)
        self.linear_end = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        labels["start_positions"] (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        labels["end_positions"] (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits_start = self.linear_start(sequence_output).squeeze(-1)
        logits_end = self.linear_end(sequence_output).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss_start = loss_fct(logits_start, labels['start_positions'])
            loss_end = loss_fct(logits_end, labels['end_positions'])
            loss = (loss_start + loss_end) / 2.0

        if not return_dict:
            output = (logits_start, logits_end) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=logits_start,
            end_logits=logits_end,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SpanExtractionModel(Model):
    """Model for span extraction.
    """
    task = 'span'

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add command line arguments."""
        group = super().add_cmdline_args(parser)
        group.add_argument("--threshold", type=float, default=0.5,
            help="Threshold for position probability. Default: 0.5.")
        group.add_argument("--model_name_or_path", default="uie-base-torch", type=str,
                           help="UIE or BERT-like model in huggingface.co or a path to model weights. "
                                "Default: 'uie-base-torch'.")
        return group

    def __init__(self, args) -> None:
        super().__init__(args)
        self.model_name_or_path = args.model_name_or_path
        self.threshold = args.threshold
        self.setup_model()
        self.setup_metrics()

    def setup_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=self.use_fast_tokenizer)
        self.model = BertForSpan.from_pretrained(self.model_name_or_path)
        if self.special_tokens:
            with open(self.special_tokens, "r") as fin:
                special_tokens = [w.strip() for w in fin]
            self._tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            print(f'add special tokens: {len(special_tokens)}')
        self.model.resize_token_embeddings(len(self._tokenizer))

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        inputs = self.create_inputs(batch)
        outputs = self.forward(inputs)
        loss = outputs.loss
        self.log("train_loss", loss)
        return {'loss': loss}

    def _common_step(self, stage: str, batch: Any) -> Dict[str, torch.Tensor]:
        inputs = self.create_inputs(batch, stage=stage)
        outputs = self.forward(inputs)
        loss = outputs.loss
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        if inputs["labels"] is not None:
            # TODO
            pass
        return {'loss': loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._common_step("test", batch)

    def predict_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        inputs = self.create_inputs(batch, stage='predict')
        outputs = self.forward(inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_probs = torch.sigmoid(start_logits).tolist()
        end_probs = torch.sigmoid(end_logits).tolist()
        start_ids_list = sputils.get_bool_ids_greater_than(
            start_probs, limit=self.threshold, return_prob=True)
        end_ids_list = sputils.get_bool_ids_greater_than(
            end_probs, limit=self.threshold, return_prob=True)

        offset_mapping = inputs['offset_mapping'].tolist()
        sentence_ids = []
        probs = []
        for start_ids, end_ids, offset_map in zip(start_ids_list, end_ids_list, offset_mapping):
            span_set = sputils.get_span(start_ids, end_ids, with_prob=True)
            sentence_id, prob = sputils.get_id_and_prob(span_set, offset_map)
            sentence_ids.append(sentence_id)
            probs.append(prob)

        raw_inputs = [{'prompt': lhs, 'text': rhs} for lhs, rhs in (zip(batch['prompt'], batch['text']))]
        preds = self._convert_ids_to_results(raw_inputs, sentence_ids, probs)

        return {'preds': preds}

    def setup_metrics(self):
        pass

    def create_inputs(self, batch: Any, stage: str = 'fit') -> Dict[str, torch.Tensor]:
        """ Write your custom inputs creating function.
        """
        def _create_labels(batch_entities):
            pass

        assert 'prompt' in batch and 'text' in batch, 'data fields unvalid'
        texts_or_text_pairs = list(zip(batch['prompt'], batch['text']))

        inputs = self._tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation='only_second',
            return_tensors='pt',
        )
        if stage == 'predict':
            inputs["labels"] = None
        else:
            inputs["labels"] = _create_labels(batch["entities"])

        if torch.cuda.is_available():
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.cuda()

        return inputs

    def _convert_ids_to_results(self, examples, sentence_ids, probs):
        """
        Convert ids to raw text in a single stage.
        """
        results = []
        for example, sentence_id, prob in zip(examples, sentence_ids, probs):
            if len(sentence_id) == 0:
                results.append([])
                continue
            result_list = []
            text = example["text"]
            prompt = example["prompt"]
            for i in range(len(sentence_id)):
                start, end = sentence_id[i]
                if start < 0 and end >= 0:
                    continue
                if end < 0:
                    start += (len(prompt) + 1)
                    end += (len(prompt) + 1)
                    result = {"text": prompt[start:end], "probability": prob[i]}
                    result_list.append(result)
                else:
                    result = {
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "probability": prob[i]
                    }
                    result_list.append(result)
            results.append(result_list)
        return results
