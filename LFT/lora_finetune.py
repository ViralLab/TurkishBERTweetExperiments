import sys, os
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_model,
    LoraConfig,
)

from datasets import load_from_disk, disable_caching

disable_caching()
from transformers import (
    AutoModelForSequenceClassification,
    MT5ForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)


class History:
    def __init__(self) -> None:
        self.epoch_history = []

    def add_epoch_history(self, epoch_hist):
        self.epoch_history.append(epoch_hist)


def to_json(dictionary, path):
    with open(path, "w") as fh:
        json.dump(dictionary, fh, indent=4)


def save_classification_report(report, path):
    to_json(report, path)


class Metric:
    def __init__(self, name="train") -> None:
        self.name = name
        self.predictions = torch.tensor([])
        self.references = torch.tensor([])

        self.epoch_history = []

    def add_batch_history(self, references, predictions):
        self.predictions = torch.concat([self.predictions, predictions], dim=0)
        self.references = torch.concat([self.references, references], dim=0)

    def compute(self) -> dict:
        return {
            "weighted_f1": f1_score(
                self.references, self.predictions, average="weighted"
            ),
            "macro_f1": f1_score(self.references, self.predictions, average="macro"),
            "acc": accuracy_score(self.references, self.predictions),
            "weighted_percision": precision_score(
                self.references, self.predictions, average="weighted"
            ),
            "weighted_recall": recall_score(
                self.references, self.predictions, average="weighted"
            ),
        }

    def get_report(self):
        return classification_report(
            self.references, self.predictions, output_dict=True
        )

    def __repr__(self) -> str:
        return f"{self.name} metric: {self.compute()}"


def load_config(config_path):
    with open(config_path, "r") as fh:
        return json.load(fh)


def tokenize_function(tokenizer, examples, max_length):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["sentence"], truncation=True, max_length=max_length)
    return outputs


def collate_fn(tokenizer, examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


def train_and_get_best_model(
    config,
    model,
    device,
    train_dataloader,
    test_dataloader,
    optimizer,
    lr_scheduler,
    fold=1,
):
    model = model.to(device)
    num_epochs = config["epochs"]
    best_f1 = -100
    best_model = None

    history = History()
    for epoch in range(num_epochs):
        epoch += 1
        # Training
        model = model.train()
        train_tqdm = tqdm(train_dataloader)
        train_tqdm.set_description(f"[Train:{fold}] Epoch {epoch}")
        train_metric = Metric("train")
        for step, batch in enumerate(train_tqdm):
            batch.to(device)
            outputs = model(**batch)

            # setting num labels to avoid index out of range error
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            predictions = outputs.logits.argmax(dim=-1)
            references = batch["labels"]
            # Cache predictions and references for metric computation.
            train_metric.add_batch_history(
                references.cpu().detach(), predictions.cpu().detach()
            )

        # Evaluation
        eval_metric = Metric("eval")
        model = model.eval()
        eval_tqdm = tqdm(test_dataloader)
        eval_tqdm.set_description(f"[Eval:{fold}] Epoch {epoch}")
        for step, batch in enumerate(eval_tqdm):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            references = batch["labels"]

            # Cache predictions and references for metric computation.
            eval_metric.add_batch_history(
                references.cpu().detach(), predictions.cpu().detach()
            )

        train_hist = train_metric.compute()
        eval_hist = eval_metric.compute()
        history.add_epoch_history({"train": train_hist, "eval": eval_hist})

        print("-" * 200)
        if eval_hist["weighted_f1"] > best_f1:
            best_f1 = eval_hist["weighted_f1"]
            best_model = model
            print("BEST MODEL UPDATED,", f"F1: {best_f1}")
        print(f"[TRAIN:{fold}] Epoch {epoch}:", train_hist)
        print(f"[EVAL:{fold}] Epoch {epoch}:", eval_hist)
        print("-" * 200)
    return best_model, history


def evaluate_best_model(model, device, data_loader, name="train", fold=1):
    model = model.to(device)
    model = model.eval()
    eval_tqdm = tqdm(data_loader)
    eval_tqdm.set_description(f"[{name.upper()}:BEST:{fold}]")
    metric = Metric(name)
    for step, batch in enumerate(eval_tqdm):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        references = batch["labels"]

        # Cache predictions and references for metric computation.
        metric.add_batch_history(references.cpu().detach(), predictions.cpu().detach())

    return metric, metric.compute(), metric.get_report()


def get_experiment_folder_path(config):
    dataset_name = config["data_path"].split("/")[-1]
    experiment_folder = (
        f"experiments/{dataset_name}_{config['model_name']}_{config['task_name']}"
    )
    print("EXPERIMENT FOLDER:", experiment_folder)
    Path(experiment_folder).mkdir(exist_ok=True)
    return experiment_folder


def get_fold_path(experiment_folder, fold):
    path = f"{experiment_folder}/fold_{fold}"
    Path(path).mkdir(exist_ok=True)
    return path


def get_peft_config(config):
    peft_config = LoraConfig(
        task_type=config["task_type"],
        target_modules=config["target_modules"],
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
    )
    return peft_config


def get_n_fold(config):
    parent_folder = config["data_path"]
    return len([f for f in os.listdir(parent_folder) if "fold" in f])


def init_model(config):
    peft_config = get_peft_config(config)

    if "t5" in config["pretrained_model_path"].lower():
        model = MT5ForConditionalGeneration.from_pretrained(
            config["pretrained_model_path"], return_dict=True
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config["pretrained_model_path"],
            return_dict=True,
            num_labels=config["num_labels"],
        )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if any(k in config["pretrained_model_path"] for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    # loading Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["pretrained_model_path"], padding_side=padding_side
    )
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    experiment_folder_path = get_experiment_folder_path(config)
    n_fold = get_n_fold(config)

    for fold in range(n_fold):
        fold += 1
        ds_name = config["data_path"].split("/")[-1]
        fold_data_path = f"{config['data_path']}/{ds_name}_fold_{fold}"
        experiment_fold_path = get_fold_path(experiment_folder_path, fold)

        # Lodading dataset
        my_dataset = load_from_disk(fold_data_path)
        num_labels = len(set(my_dataset["train"]["labels"]))
        config["num_labels"] = num_labels

        def lower_down(example):
            example["sentence"] = example["sentence"].lower()
            return example

        my_dataset = my_dataset.map(lower_down)

        tokenized_datasets = my_dataset.map(
            lambda x: tokenize_function(tokenizer, x, config["input_len"]),
            batched=True,
            remove_columns=["sentence"],
        )

        # Instantiate dataloaders.
        train_dataloader = DataLoader(
            tokenized_datasets["train"],
            shuffle=True,
            collate_fn=lambda x: collate_fn(tokenizer, x),
            batch_size=config["train_batch_size"],
        )
        test_dataloader = DataLoader(
            tokenized_datasets["test"],
            shuffle=False,
            collate_fn=lambda x: collate_fn(tokenizer, x),
            batch_size=config["test_batch_size"],
        )

        model = init_model(config)

        optimizer = AdamW(params=model.parameters(), lr=config["learning_rate"])

        # Instantiate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0.06 * (len(train_dataloader) * config["epochs"]),
            num_training_steps=(len(train_dataloader) * config["epochs"]),
        )

        best_model, history = train_and_get_best_model(
            config,
            model,
            device,
            train_dataloader,
            test_dataloader,
            optimizer,
            lr_scheduler,
            fold=fold,
        )

        # Save best model
        best_model.save_pretrained(experiment_fold_path + "/best_model")

        # Save history
        to_json(history.epoch_history, experiment_fold_path + "/history.json")

        train_metric, _, best_train_report = evaluate_best_model(
            best_model, device, train_dataloader, name="train", fold=fold
        )
        test_metric, _, best_eval_report = evaluate_best_model(
            best_model, device, test_dataloader, name="eval", fold=fold
        )

        save_classification_report(
            best_train_report, experiment_fold_path + "/best_train_report.json"
        )
        save_classification_report(
            best_eval_report, experiment_fold_path + "/best_test_report.json"
        )

        np.save(
            experiment_fold_path + "/best_train_predictions",
            train_metric.predictions.cpu().numpy(),
        )
        np.save(
            experiment_fold_path + "/best_test_predictions",
            test_metric.predictions.cpu().numpy(),
        )


if __name__ == "__main__":
    config = load_config(sys.argv[1])
    main(config)
