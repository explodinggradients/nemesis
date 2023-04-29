from turtle import forward
from dataclasses import dataclass
from transformers import GPTNeoXConfig, GPTNeoXPreTrainedModel, GPTNeoXForCausalLM, GPTNeoXModel
from torch import nn
import torch
from transformers.utils import ModelOutput


@dataclass
class GPTNeoxRMOuptput(ModelOutput):
        """
        Reward Model Output
        """
        logits: torch.FloatTensor = None

class GPTNeoXRM(GPTNeoXPreTrainedModel):
    """
    """
    def __init__(
            self,
            config,
    ):
        super().__init__(config)
        self.gpt_neox = GPTNeoXModel(config)
        self.out_layer = nn.Linear(config.hidden_size, 1)

    def forward(
            self,
            input_ids,
            attention_mask,
            **kwargs,

    ):
        return_dict = kwargs.get("return_dict") if kwargs.get("return_dict") is not None else self.config.use_return_dict
        outputs = self.gpt_neox(
            input_ids,
            attention_mask,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = outputs[0]
        if attention_mask is None:
             hidden_states = hidden_states.mean(dim=1)
        else:
             hidden_states = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
        print(hidden_states.shape)
        lm_logits = self.out_layer(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]
        
        return GPTNeoxRMOuptput(
             logits=lm_logits
        )    

 

