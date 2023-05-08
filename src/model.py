from dataclasses import dataclass
from turtle import hideturtle
from transformers import (
    GPTNeoXConfig,
    GPTNeoXPreTrainedModel,
    GPTNeoXModel,
    AutoModelForSequenceClassification,
    AutoConfig,
)
from torch import nn
import torch
from transformers.utils import ModelOutput


@dataclass
class GPTNeoxRMOuptput(ModelOutput):
    """
    Reward Model Output
    """

    logits: torch.FloatTensor = None

class GPTNeoXConfigRM(GPTNeoXConfig):
    model_type = "rm_gptneox_config"
    def __init__(
        self,
        pooling = "last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pooling = pooling


class GPTNeoXRM(GPTNeoXPreTrainedModel):
    config_class = GPTNeoXConfigRM
    """ 
    Reward Model
    """

    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.gpt_neox = GPTNeoXModel(config)
        self.pooling = config.pooling
        hidden_size = config.hidden_size if self.pooling != "mean-max" else config.hidden_size * 2
        self.out_layer = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        **kwargs,
    ):
        return_dict = (
            kwargs.get("return_dict")
            if kwargs.get("return_dict") is not None
            else self.config.use_return_dict
        )
        outputs = self.gpt_neox(
            input_ids,
            attention_mask,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = outputs[0]
        if self.pooling == "mean":
            if attention_mask is None:
                hidden_states = hidden_states.mean(dim=1)
            else:
                hidden_states = (hidden_states * attention_mask.unsqueeze(-1)).sum(
                    dim=1
                ) / attention_mask.sum(dim=1).unsqueeze(-1)
        elif self.pooling == "last":
            if attention_mask is None:
                hidden_states = hidden_states[:,-1,:]
            else:
                last_idx = attention_mask.cumsum(1).argmax(1)
                last_idx = last_idx.view(-1,1,1).expand(-1,1,hidden_states.size(-1))
                hidden_states = torch.gather(hidden_states,1,last_idx).squeeze(1)
        elif self.pooling == "mean-max":
            if attention_mask is None:
                mean, max = hidden_states.mean(dim=1), hidden_states.max(dim=1).values
                hidden_states = torch.cat([mean,max],1)
            else:
                mean = (hidden_states * attention_mask.unsqueeze(-1)).sum(
                    dim=1
                ) / attention_mask.sum(dim=1).unsqueeze(-1)
                max = (hidden_states * attention_mask.unsqueeze(-1)).max(
                    dim=1
                ).values
                hidden_states = torch.cat([mean, max], 1)
        else:
            raise ValueError(f"invalid pooling {self.pooling}")

        lm_logits = self.out_layer(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return GPTNeoxRMOuptput(logits=lm_logits)


AutoConfig.register("rm_gptneox_config", GPTNeoXConfigRM)
AutoModelForSequenceClassification.register(GPTNeoXConfigRM, GPTNeoXRM)