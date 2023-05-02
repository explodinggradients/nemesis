from collections import defaultdict
from sre_parse import SPECIAL_CHARS
from typing import Any
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
import torch
from utils import SPECIAL_TOKENS

@dataclass
class HFSummary(Dataset):
    name = "openai/summarize_from_feedback"
    
    def __init__(self, split:str="train"):
        super().__init__()
        self.split = split
        dataset = load_dataset(self.name,"axis",split=split)
        self.data_dict = self.prepare_axis(dataset)
        self.postids = list(self.data_dict.keys())

    def prepare_axis(self, dataset):

        data_dict = defaultdict(dict)
        for item in dataset:
            if item["summary"]["axes"].get("overall") is not None:
                postid = item["info"]["id"]
                summary = {k:item["summary"][k] for k in ["text","axes"]}
                if postid not in data_dict.keys():
                    instruction = "summarize: " + (item["info"]["post"]  or item["info"]["article"])
                    data_dict[postid].update({"post":instruction,"summaries":[summary]})
                else:
                    data_dict[postid]["summaries"].append(summary)
        
        return data_dict
    
    def __len__(self):
        return len(self.postids)
    
    def __getitem__(self,idx):
        post,summaries = self.data_dict[self.postids[idx]].values()
        summaries = sorted(summaries,key=lambda x:x['axes']['overall'],reverse=True)
        summaries = [item["text"].strip() for item in summaries]
        return post, summaries
    
@dataclass
class RMDataCollator:
    tokenizer:PreTrainedTokenizer
    max_length:int=512


    def format_example(self, example,eos,prompt=False):
        sp_token = SPECIAL_TOKENS["prompter"] if prompt else SPECIAL_TOKENS["assistant"]
        return "{}{}{}".format(sp_token,example,eos)

    def process_example(self,example):

        trunc_len = 0
        eos = self.tokenizer.eos_token
        prefix,outputs = example
        prefix = self.format_example(example,eos,prompt=True)
        outputs = [self.format_example(output,eos) for output in outputs]

        prefix_tokens = self.tokenizer.encode(prefix)
        input_ids, attention_masks = [],[]
        for output in outputs:
            out_tokens = self.tokenizer.encode(output,)
            if len(prefix_tokens) + len(out_tokens) > self.max_length:
                trunc_len = max(0,len(prefix_tokens) + len(out_tokens) - self.max_length)
            prefix_tokens = prefix_tokens[trunc_len:]
            out_tokens = prefix_tokens + out_tokens
            out_tokens = out_tokens[:self.max_length]
            pad_len = self.max_length - len(out_tokens)
            attn_masks = [1] * len(out_tokens) + [0] * pad_len
            out_tokens += [self.tokenizer.pad_token_id] * pad_len
            input_ids.append(out_tokens)
            attention_masks.append(attn_masks)
        return input_ids, attention_masks
    
    def __call__(self,examples):
        
        batch_k_lens = [0]
        input_ids, attention_masks = [],[]
        for i,example in enumerate(examples):
            inp_ids,attn_masks = self.process_example(example)
            input_ids.extend(inp_ids)
            attention_masks.extend(attn_masks)
            batch_k_lens.append(batch_k_lens[i]+len(inp_ids))

        return {
            "input_ids":torch.tensor(input_ids),
            "attention_mask":torch.tensor(attention_masks),
            "k_lens":batch_k_lens
        }
            


        