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
    destination_path: Path = Path("data/stack_exchange_rm"), 
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


    print(f"train has {len(_train_set):,} samples")
    print(f"val has {len(_test_set):,} samples")

    print("Processing train split ...")
    train_set = []
    for i,sample in tqdm(enumerate(_train_set)):
        train_set.append(prepare_sample(sample, tokenizer, max_seq_length, mask_inputs))
        if i>10000:
            break

    torch.save(train_set, destination_path / "train.pt")
    

    print("Processing test split ...")
    test_set = []
    for i,sample in tqdm(enumerate(_test_set)):
        test_set.append(prepare_sample(sample, tokenizer, max_seq_length, mask_inputs))
        if i>2000:
            break
    
    torch.save(test_set, destination_path / "test.pt")


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

    question = example.get("question")
    response_j = example.get("response_j")
    response_k = example.get("response_k")


                           
    response_j = "Question: " + question + "\n\nAnswer: " + response_j
    response_k = "Question: " + question + "\n\nAnswer: " + response_k

    print(response_j, type(response_j))
    print(response_k, type(response_k))

    encoded_response_j = tokenize(tokenizer, response_j, max_length=max_length, eos=False)
    encoded_response_k = tokenize(tokenizer, response_k, max_length=max_length, eos=False)

    return {**example, "input_ids_j": encoded_response_j, "input_ids_k": encoded_response_k}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)



if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
