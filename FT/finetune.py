import pandas as pd
import numpy as np
import sys, json, joblib
from pathlib import Path

import os

# Model Training
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from Model.trainer import Trainer
from Model.model import ClfModel
from datasets import load_from_disk, disable_caching

disable_caching()


def load_config(config_path):
    with open(config_path, "r") as fh:
        return json.load(fh)


def save_meta_data(meta_data, path):
    joblib.dump(meta_data, path + "/meta.bin")


def get_experiment_folder_path(config):
    dataset_name = config["data_path"].split("/")[-1]
    print(config["data_path"].split("/"))
    experiment_folder = (
        f"experiments/{dataset_name}_{config['model_name']}_{config['task_name']}"
    )
    print(experiment_folder)
    Path(experiment_folder).mkdir(exist_ok=True)
    return experiment_folder


def save_fold_indices(train_indices, test_indices, path):
    Path(path).mkdir(exist_ok=True)
    np.save(f"{path}/train_indices.npy", train_indices)
    np.save(f"{path}/test_indices.npy", test_indices)


def to_json(dictionary, path):
    with open(path, "w") as fh:
        json.dump(dictionary, fh, indent=4)


def get_n_fold(config):
    parent_folder = f"{config['data_path']}"
    return len([f for f in os.listdir(parent_folder) if "fold" in f])


def calculate_loss_weights(y_train, labels2idx):
    y = np.array(y_train)
    weights = np.ones(len(labels2idx))
    for label, index in labels2idx.items():
        weights[index] = 1 / (y == index).sum()
    weights /= np.min(weights)
    return weights


def collate_fn(tokenizer, examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


def tokenize_function(tokenizer, examples, max_length):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["sentence"], truncation=True, max_length=max_length)
    return outputs


def get_fold_path(experiment_folder, fold):
    path = f"{experiment_folder}/fold_{fold}"
    Path(path).mkdir(exist_ok=True)
    return path


if __name__ == "__main__":
    config = load_config(sys.argv[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])

    experiment_folder_path = get_experiment_folder_path(config)
    n_fold = get_n_fold(config)
    for fold in range(n_fold):
        # Lodading dataset
        ds_name = config["data_path"].split("/")[-1]
        fold_data_path = f"{config['data_path']}/{ds_name}_fold_{fold+1}"

        experiment_fold_path = get_fold_path(experiment_folder_path, fold + 1)
        config["output_dir"] = experiment_fold_path

        my_dataset = load_from_disk(fold_data_path)

        num_labels = len(set(my_dataset["train"]["labels"]))
        config["num_labels"] = num_labels
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

        label_encoder = joblib.load(config["data_path"] + "/meta.bin")
        labels2index = dict(
            zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
        )

        loss_weights = calculate_loss_weights(
            tokenized_datasets["train"]["labels"], labels2index
        )

        updated_labels2index = dict()
        for i, k in labels2index.items():
            k = int(k)
            if type(i) == np.int64:
                i = int(i)
            updated_labels2index[i] = int(k)
        config["labels2idx"] = updated_labels2index
        config["loss_weights"] = loss_weights.tolist()

        to_json(config, f"{config['output_dir']}/config.json")

        model = ClfModel(config)

        trainer = Trainer(
            config=config,
            model=model,
            train_dataloader=train_dataloader,
            validation_dataloader=test_dataloader,
            loss_weights=config["loss_weights"],
        )
        print(f"\nFOLD {fold+1}\n")
        hist = trainer.fit(fold_n=fold)

        to_json(hist, f"{config['output_dir']}/history.json")

        train_preds, train_cm, train_acc, train_report = trainer.evaluate(
            model, train_dataloader
        )
        test_preds, test_cm, test_acc, test_report = trainer.evaluate(
            model, test_dataloader
        )

        to_json(train_report, f"{config['output_dir']}/train_report.json")
        to_json(test_report, f"{config['output_dir']}/test_report.json")

        np.save(f"{config['output_dir']}/train_preds.npy", train_preds.cpu())
        np.save(f"{config['output_dir']}/test_preds.npy", test_preds.cpu())

        #####################  BEST MODEL #####################

        # load the best model
        trainer.load_model(f"{config['output_dir']}/best_{config['model_name']}.pt")

        train_preds, train_cm, train_acc, train_report = trainer.evaluate(
            model, train_dataloader
        )
        test_preds, test_cm, test_acc, test_report = trainer.evaluate(
            model, test_dataloader
        )

        to_json(train_report, f"{config['output_dir']}/best_train_report.json")
        to_json(test_report, f"{config['output_dir']}/best_test_report.json")

        np.save(f"{config['output_dir']}/best_train_preds.npy", train_preds.cpu())
        np.save(f"{config['output_dir']}/best_test_preds.npy", test_preds.cpu())
