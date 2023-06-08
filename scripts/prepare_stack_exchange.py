"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm

from datasets import load_dataset



DATASET_NAME = "lvwerra/stack-exchange-paired"
IGNORE_INDEX = -1

def prepare(
    destination_path: Path = Path("data/stack_exchange"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    test_split_size: float = 0.005,
    max_seq_length: int = 256,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    dataset_name: str = DATASET_NAME
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    
    destination_path.mkdir(parents=True, exist_ok=True)

    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)
    

    dataset = load_dataset(
        dataset_name,
        data_dir="data/finetune",
        split="train",
    )
    # Partition the dataset into train and test
    dataset = dataset.train_test_split(test_size=test_split_size, seed=seed)
    _train_set = dataset["train"]
    _test_set = dataset["test"]


    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = []
    for i,sample in tqdm(enumerate(_train_set)):
        train_set.append(prepare_sample(convert_stack_exchange_to_alpaca_format(sample), tokenizer, max_seq_length, mask_inputs))
        if i>50000:
            break

    torch.save(train_set, destination_path / "train.pt")
    

    print("Processing test split ...")
    test_set = []
    for i,sample in tqdm(enumerate(_test_set)):
        test_set.append(prepare_sample(convert_stack_exchange_to_alpaca_format(sample), tokenizer, max_seq_length, mask_inputs))
        if i>2000:
            break
    
    torch.save(test_set, destination_path / "test.pt")



def convert_stack_exchange_to_alpaca_format(input_x):
    instruction = input_x.get("question","")
    output = input_x.get("response_j")
    return {"instruction":instruction, "input": "", "output": output}

def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True):
    """Processes a single sample.
    
    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The input text is formed as a single message including all
    the instruction, the input (optional) and the response.
    The label/target is the same message but can optionally have the instruction + input text
    masked out (mask_inputs=True).

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    return {**example, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
