from transformers import PreTrainedTokenizer
import torch
from dataclasses import dataclass

from utils import SPECIAL_TOKENS


@dataclass
class RMDataCollator:
    tokenizer: PreTrainedTokenizer
    max_length: int = 512

    def format_prefix(self, prompts, eos):

        prompts = ["{}{}{}".format(SPECIAL_TOKENS["prompter"] if i%2==0 else SPECIAL_TOKENS["assistant"], prompt, eos) for i,prompt in enumerate(prompts)]
        return "".join(prompts)
    
    def format_suffix(self, answer, eos):
        return "{}{}{}".format(SPECIAL_TOKENS["assistant"], answer, eos)

    def process_example(self, example):
        trunc_len = 0
        eos = self.tokenizer.eos_token
        prefix, outputs = example
        prefix = self.format_prefix(prefix, eos)
        outputs = [self.format_suffix(output, eos) for output in outputs]
        print(prefix,outputs)
        prefix_tokens = self.tokenizer.encode(prefix)
        input_ids, attention_masks = [], []
        for output in outputs:
            out_tokens = self.tokenizer.encode(
                output,
            )
            if len(prefix_tokens) + len(out_tokens) > self.max_length:
                trunc_len = max(
                    0, len(prefix_tokens) + len(out_tokens) - self.max_length
                )
            prefix_tokens = prefix_tokens[trunc_len:]
            out_tokens = prefix_tokens + out_tokens
            out_tokens = out_tokens[: self.max_length]
            pad_len = self.max_length - len(out_tokens)
            attn_masks = [1] * len(out_tokens) + [0] * pad_len
            out_tokens += [self.tokenizer.pad_token_id] * pad_len
            input_ids.append(out_tokens)
            attention_masks.append(attn_masks)
        return input_ids, attention_masks

    def __call__(self, examples):
        batch_k_lens = [0]
        input_ids, attention_masks = [], []
        for i, example in enumerate(examples):
            inp_ids, attn_masks = self.process_example(example)
            input_ids.extend(inp_ids)
            attention_masks.extend(attn_masks)
            batch_k_lens.append(batch_k_lens[i] + len(inp_ids))

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks),
            "k_lens": batch_k_lens,
        }
