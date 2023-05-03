from collections import defaultdict
from torch.utils.data import Dataset
from datasets import load_dataset
import re
from typing import Union, List
from omegaconf import OmegaConf 

class HFSummary(Dataset):
    name = "openai/summarize_from_feedback"
    
    def __init__(self, split:Union[List[str], str]="train"):
        super().__init__()
        if isinstance(split,str):
            split = [split]
        self.split = OmegaConf.to_object(split)
        dataset = load_dataset(self.name,"axis",split=self.split)
        self.data_dict = self.prepare_axis(dataset)
        self.postids = list(self.data_dict.keys())

    def prepare_axis(self, dataset):

        data_dict = defaultdict(dict)
        for data in dataset:
            for item in data:
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
        dedup_dict = {item["axes"]["overall"]:item["text"] for item in summaries}
        summaries = {key:val for val,key in dedup_dict.items()}
        summaries = list(summaries.keys())
        return post, summaries

class WebGPT:
    name = "openai/webgpt_comparisons"

    def __init__(self, split:str="train"):
        super().__init__()
        self.split = split
        dataset = load_dataset(self.name, split=self.split)
        self.dataset_dict = defaultdict(dict)
        for item in dataset:
            post_id = item["question"]["id"]
            if post_id not in self.dataset_dict.keys():
                self.dataset_dict[post_id] = {"full_text": item["question"]["full_text"],
                                              "answers":[]}
                if item["score_0"] > 0:
                    answers = [item["answer_0"],item["answer_1"]]
                elif item["score_0"] < 0:
                    answers = [item["answer_1"],item["answer_0"]]
                else:
                    answers = []
                answers = [re.sub(r"\[\d+\]","",answer) for answer in answers]
                answers = [".".join([sent.strip() for sent in answer.split('.')]) for answer in answers]
                if answers:
                    self.dataset_dict[post_id]["answers"].extend(answers)
                else:
                    _ = self.dataset_dict.pop(post_id)

        self.post_ids = list(self.dataset_dict.keys())

    def __len__(self):
        return len(self.post_ids)

    def __getitem__(self, idx):
        question, answers = self.dataset_dict[self.post_ids[idx]].values()
        return question, answers
    




        