import torch
from tokenizers import pre_tokenizers
from torch.utils.data import ConcatDataset, random_split
from transformers import AutoTokenizer

from dataset import AnthropicRLFH, HFSummary, WebGPT

SPECIAL_TOKENS = {"prompter": "|prompter|", "assistant": "|assistant|"}
generator = torch.Generator().manual_seed(42)


def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    if hasattr(config, "per_digit_tokens") and config.per_digit_tokens:
        tokenizer._tokenizer.pre_processor = pre_tokenizers.Digits(True)

    if config.special_tokens:
        special_tokens = {
            "pad_token": config.special_tokens.pad_token,
            "eos_token": config.special_tokens.eos_token,
            "sep_token": config.special_tokens.sep_token,
        }
        tokenizer.add_special_tokens(special_tokens)

    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(SPECIAL_TOKENS.values())}
    )

    return tokenizer


def get_single_dataset(name, **kwargs):
    if name == "hf_summary":
        dataset = HFSummary(**kwargs)
    elif name == "webgpt":
        dataset = WebGPT(**kwargs)
    elif name == "AnthropicRLHF":
        dataset = AnthropicRLFH(**kwargs)
    else:
        raise ValueError(f"Invalid dataset name {name}")

    return dataset


def prepare_datasets(config):
    dataset_list = []
    for dataset in config.datasets:
        name = list(dataset.keys())[0]
        kwargs = dataset[name]
        dataset_list.append(get_single_dataset(name, **kwargs))

    dataset = ConcatDataset(dataset_list)
    train_dataset, valid_dataset = random_split(
        dataset,
        [1 - config.validation_size, config.validation_size],
        generator=generator,
    )
    return train_dataset, valid_dataset
