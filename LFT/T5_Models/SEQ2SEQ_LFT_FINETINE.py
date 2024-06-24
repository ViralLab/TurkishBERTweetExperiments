import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from datasets import load_from_disk
import datasets as ds

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    GPTJForQuestionAnswering,
)
from peft import (
    PeftModel,
    PeftConfig,
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

from pathlib import Path


device = "cuda:0"
model_save = True
BATCH_SIZE = 16
model_id = "google/mt5-large"  # TURNA
ds_name = "DS_PATH"
initial_learning_rate = 8e-5
patience = 1000
decay_factor = 0.8


def get_tokenizer(model_id="google/mt5-large"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=512)
    return tokenizer


def get_model(model_id="google/mt5-large"):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.01,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    # model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    return model


def separate_input_output(example):
    label = example["text"].split("A:")[-1].strip()
    text = example["text"].split("A:")[0].strip()
    return {"text": text, "label": label}


# function to tokenize the input and response
def input_tokenizer(examples, tokenizer):
    questions = examples["text"]
    responses = examples["label"]
    inputs = [question for question in questions]
    model_inputs = tokenizer(inputs, padding=False, truncation=True)

    labels = tokenizer(responses, padding=False, truncation=True).input_ids
    model_inputs["labels"] = labels

    return model_inputs


# function to tokenize the input and response
def input_tokenizer_hs(examples, tokenizer):
    questions = examples["text"]
    responses = examples["label"]

    inputs = [question for question in questions]
    model_inputs = tokenizer(inputs, padding=False, truncation=True)

    labels = tokenizer(responses, padding=False, truncation=True).input_ids
    model_inputs["labels"] = labels

    return model_inputs


def get_trainer(tokenizer, model, train_ds, test_ds, output_dir):
    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        num_train_epochs=20,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        fp16=False,
        save_total_limit=1,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
        ),
    )

    model.config.use_cache = False

    return trainer


# PREPAREING DATASET
my_dataset = load_from_disk(f"DS_PATH_ROOT/{ds_name}")
my_dataset = my_dataset.map(separate_input_output)


tokenizer = get_tokenizer(model_id)

# maping the above function to tokenize the dataset, removed the unnecessary features
tokenized_my_dataset = my_dataset.map(
    lambda x: input_tokenizer_hs(x, tokenizer),
    batched=True,
    load_from_cache_file=False,
    remove_columns=my_dataset.column_names,
)


# setting to format to torch as using torch for training
tokenized_my_dataset.set_format("torch")

skf = StratifiedKFold(n_splits=10, random_state=139, shuffle=True)
labels = my_dataset["label"]

for i, (train_index, test_index) in tqdm(
    enumerate(skf.split(np.zeros(len(labels)), labels)), total=10
):

    train_ds = tokenized_my_dataset.select(train_index)
    test_ds = tokenized_my_dataset.select(test_index)

    # init Model

    model = get_model(model_id)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    # train dataloader
    train_dataloader = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )

    # test dataloader
    validation_dataloader = DataLoader(test_ds, batch_size=1, collate_fn=data_collator)
    Path(f"./output{model_id}_10FOLD/{ds_name}/").mkdir(parents=True, exist_ok=True)
    output_dir = f"./output{model_id}_10FOLD/{ds_name}/{model_id}-{ds_name}-fold-{i}"

    trainer = get_trainer(tokenizer, model, train_ds, test_ds, output_dir)

    trainer.train()
