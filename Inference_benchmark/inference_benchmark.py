"""
This script is for benchmarking the inference time of various models.   
"""

import time
import json
import pandas as pd
from collections import defaultdict
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from datasets import Dataset
from torch.utils.data import DataLoader
import datasets

datasets.disable_caching()


from tqdm import tqdm

DATASET_PATH = "data.jsonl"
model_id = "MODEL_ID"
N_EXPERIMENTS = 100
model_input_len = 128
index = 13
number_of_samples = [2**i for i in range(index)][::-1]
MAX_NUMBER_OF_SAMPLES = number_of_samples[0]

POSSIBLE_INDEX = 6
BEST_BATCH_SIZE = 2**POSSIBLE_INDEX
DEVICE = "cuda:0"


def tokenize_function(tokenizer, examples, max_length):
    outputs = tokenizer(examples["text"], truncation=True, max_length=max_length)
    return outputs


def collate_fn(tokenizer, examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


df = pd.read_json(DATASET_PATH, lines=True, orient="records")
texts = df["p_text"].tolist()[:MAX_NUMBER_OF_SAMPLES]
my_dataset = Dataset.from_dict(
    {
        "text": texts,
    }
)


tokenizer = AutoTokenizer.from_pretrained(model_id)
if "llama" in model_id.lower():
    tokenizer.pad_token_id = 0
    quantization_config = None
    if "70" in model_id.lower():
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        quantization_config=quantization_config,
    )
else:
    model = AutoModel.from_pretrained(model_id)
model = model.eval()

if not ("70" in model_id.lower() and "llama" in model_id.lower()):
    model = model.to(DEVICE)

tokenized_datasets = my_dataset.map(
    lambda x: tokenize_function(tokenizer, x, model_input_len),
    batched=True,
    remove_columns=["text"],
)


with torch.no_grad():
    n_samples2times = defaultdict(list)
    for n_samples in tqdm(number_of_samples):
        sub_ds_to_infer = tokenized_datasets.select(range(n_samples))
        dataloader = DataLoader(
            sub_ds_to_infer,
            shuffle=False,
            collate_fn=lambda x: collate_fn(tokenizer, x),
            batch_size=BEST_BATCH_SIZE,
        )
        for i in range(N_EXPERIMENTS):
            start_time = time.time()
            for batch in dataloader:
                # batch.pop("token_type_ids")
                batch.to(DEVICE)
                if "turna" in model_id.lowre() or "mt5" in model_id.lower():
                    outputs = model(decoder_input_ids=batch["input_ids"], **batch)
                else:
                    outputs = model(**batch)
            end_time = time.time()
            n_samples2times[n_samples].append(end_time - start_time)

# save the results to json
if "albert" in model_id:
    model_name = "loodos_albert-base-turkish-uncased"
else:
    model_name = model_id.replace("/", "_")

output_dir = "benchmark_results_128"
output_path = f"{output_dir}/{model_name}_inference_benchmark.json"
with open(output_path, "w") as f:
    json.dump(n_samples2times, f)
