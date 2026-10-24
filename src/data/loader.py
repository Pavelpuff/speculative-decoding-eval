from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def load_wikitext_split(split: str, tokenizer: PreTrainedTokenizerBase) -> list[list[int]]:
    """
    Loads WikiText-2 split and encodes the text into token IDs.

    Args:
        split: Dataset split ('train', 'test', 'validation').
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        A list of encoded token ID sequences.
    """
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    examples = [row["text"] for row in ds if row["text"].strip()]
    tokenized_examples = [tokenizer.encode(text, add_special_tokens=False) for text in examples]
    return tokenized_examples