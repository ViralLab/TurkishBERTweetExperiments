import numpy as np
import torch
import Model.callbacks as callbacks
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from Model.model import ClfModel


class Trainer:
    def __init__(
        self,
        config,
        model,
        train_dataloader,
        validation_dataloader,
        loss_weights,
    ) -> None:
        self.config = config

        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

        self.loss_weights = loss_weights

    def fit(self, fold_n) -> dict:
        best_loss = np.inf
        best_acc = -np.inf
        early_stopper = callbacks.EarlyStopper(
            patience=self.config["early_stop_patience"],
            min_delta=self.config["early_stop_min_delta"],
        )
        train_loop_pbar = tqdm(range(self.config["epochs"]), leave=True)
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        for epoch in train_loop_pbar:
            train_loop_pbar.set_description(f"Epoch {epoch}/{self.config['epochs']}")
            train_loss, train_acc = self.train_model_one_epoch(
                self.train_dataloader,
                self.model,
                self.optimizer,
                self.device,
                self.scheduler,
            )
            validatoin_loss, validation_acc = self.evaluate_model(
                self.validation_dataloader,
                self.model,
                self.device,
            )

            train_losses.append(train_loss)
            test_losses.append(validatoin_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(validation_acc)

            # fmt: off
            print(f"\n[FOLD:{fold_n}:{epoch}] train Loss = {train_loss:.4f} ---- train Acc = {train_acc:.4f}")
            print(f"[FOLD:{fold_n}:{epoch}] validatoin Loss = {validatoin_loss:.4f}  ---- validatoin Acc = {validation_acc:.4f}")
            # fmt: on

            if best_acc < validation_acc:
                best_acc = validation_acc

            if validatoin_loss < best_loss:
                best_loss = validatoin_loss
                self.save_model(
                    f"{self.config['output_dir']}/best_{self.config['model_name']}.pt"
                )
                print("Best Model Saved!", f"accuracy:{validation_acc:.4f}")

            # early stopper
            if early_stopper.early_stop(validatoin_loss):
                print(f"Training Stoped at Epoch {epoch}")
                break
        self.save_model(
            f"{self.config['output_dir']}/last_{self.config['model_name']}.pt"
        )

        return {
            "train_loss": train_losses,
            "test_loss": test_losses,
            "train_accuracy": train_accuracies,
            "test_accuracy": test_accuracies,
        }

    def train_model_one_epoch(self, data_loader, model, optimizer, device, scheduler):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        get_accuracy = callbacks.Accuracy()
        pbar = tqdm(data_loader, total=len(data_loader), leave=True)
        pbar.set_description("Training")
        for data in pbar:
            for k, v in data.items():
                data[k] = v.to(device)
            if self.loss_weights:
                data["loss_weights"] = torch.FloatTensor(self.loss_weights).to(device)

            optimizer.zero_grad()
            if "mt5" in self.config["pretrained_model_path"]:
                logits, loss = model(decoder_input_ids=data["input_ids"], **data)
            else:
                logits, loss = model(**data)
            logits, loss = model(**data)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            y_hat = torch.argmax(logits, dim=-1)
            if self.config["task_name"] == "ner":
                epoch_acc += get_accuracy(data["target_tag"], y_hat).item()
            else:
                epoch_acc += get_accuracy(data["labels"], y_hat).item()

        epoch_loss = epoch_loss / len(data_loader)
        epoch_acc = epoch_acc / len(data_loader)

        return epoch_loss, epoch_acc

    def evaluate_model(self, data_loader, model, device):
        model.eval()
        final_loss = 0
        final_acc = 0
        get_accuracy = callbacks.Accuracy()
        pbar = tqdm(data_loader, total=len(data_loader), leave=True)
        pbar.set_description("Evaluation")
        for data in pbar:
            for k, v in data.items():
                data[k] = v.to(device)
            if self.loss_weights:
                data["loss_weights"] = torch.FloatTensor(self.loss_weights).to(device)

            if "mt5" in self.config["pretrained_model_path"]:
                logits, loss = model(decoder_input_ids=data["input_ids"], **data)
            else:
                logits, loss = model(**data)

            final_loss += loss.item()
            y_hat = torch.argmax(logits, dim=-1)

            if self.config["task_name"] == "ner":
                final_acc += get_accuracy(data["target_tag"], y_hat).item()
            else:
                final_acc += get_accuracy(data["labels"], y_hat).item()

        final_loss = final_loss / len(data_loader)
        final_acc = final_acc / len(data_loader)

        return final_loss, final_acc

    def get_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_parameters,
            lr=self.config["learning_rate"],
            no_deprecation_warning=True,
        )
        return optimizer

    def get_scheduler(self):
        num_train_steps = int(len(self.train_dataloader) * self.config["epochs"])
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps,
        )

        return scheduler

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def evaluate(self, model, data_loader):
        model.eval()

        pbar = tqdm(data_loader, total=len(data_loader), leave=True)
        pbar.set_description("Final Evaluation")
        y_pred = None
        y_true = None
        for data in pbar:
            for k, v in data.items():
                data[k] = v.to(self.device)

            if self.loss_weights:
                data["loss_weights"] = torch.FloatTensor(self.loss_weights).to(
                    self.device
                )

            logits, loss = None, None
            if "mt5" in self.config["pretrained_model_path"]:
                logits, loss = model(decoder_input_ids=data["input_ids"], **data)
            else:
                logits, loss = model(**data)

            if y_pred is None:
                y_pred = logits
            else:
                y_pred = torch.cat([y_pred, logits], dim=0)

            if y_true is None:
                if self.config["task_name"] == "ner":
                    y_true = data["target_tag"]
                else:
                    y_true = data["labels"]
            else:
                if self.config["task_name"] == "ner":
                    y_true = torch.cat([y_true, data["target_tag"]], dim=0)
                else:
                    y_true = torch.cat([y_true, data["labels"]], dim=0)

        y_pred = y_pred.cpu().detach()
        y_hat = torch.argmax(y_pred, dim=-1)
        cm = callbacks.ConfusionMatrix()(y_true.cpu(), y_hat.cpu())
        accuracy = callbacks.Accuracy()(y_true.cpu(), y_hat.cpu())

        report = callbacks.ClassificationReport()(
            y_true.cpu().detach(),
            y_hat.cpu().detach(),
            target_names=list(self.config["labels2idx"].keys()),
        )

        return y_hat, cm, accuracy, report

    def load_model(self, model_path):
        tmp_model = ClfModel(self.config)
        tmp_model.load_state_dict(torch.load(model_path))
        tmp_model.eval()

        self.model = tmp_model
