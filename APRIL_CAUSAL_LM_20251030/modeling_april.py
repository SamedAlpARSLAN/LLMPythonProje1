import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import april_model


class AprilForCausalLM(PreTrainedModel):
    config_class = None

    def __init__(self, config):
        super().__init__(config)

        if not hasattr(config, "num_hidden_layers") and hasattr(config, "num_layers"):
            config.num_hidden_layers = config.num_layers

        self.model = april_model.AprilModel(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            context_length=config.context_length,
            num_layers=config.num_layers,
            device=getattr(config, "device", "cpu"),
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        **kwargs
    ) -> CausalLMOutputWithCrossAttentions:

        logits = self.model(input_ids)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        **kwargs
    ):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "token_type_ids": token_type_ids,
        }

    def tie_weights(self):
        return


from configuration_april import AprilConfig
AprilForCausalLM.config_class = AprilConfig
