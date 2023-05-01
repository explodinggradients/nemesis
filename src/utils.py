from transformers import AutoTokenizer


SPECIAL_TOKENS = {
    "prompter":"|prompter|",
    "assistant":"|assistant|"
}


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