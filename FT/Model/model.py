from transformers import AutoModel
import torch
import torch.nn as nn


def loss_fn(y_hat, y, loss_weights=None):
    if loss_weights is None:
        lfn = nn.CrossEntropyLoss()
    else:
        lfn = nn.CrossEntropyLoss(weight=loss_weights)
    return lfn(y_hat, y.flatten())


def loss_fn_ner(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_logits = output.view(-1, num_labels)
    active_loss = mask.view(-1) == 1
    active_labels = torch.where(
        active_loss, target.view(-1), torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss


class ClfModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super(ClfModel, self).__init__()
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(
            self.config["pretrained_model_path"],
            from_flax=self.config["from_flax"],
            return_dict=True,
        )

        # freezing the layers of the pretrained_model
        for layer_name, param in self.pretrained_model.named_parameters():
            param.requires_grad = False

        self.hidden_size = self.pretrained_model.config.hidden_size
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.hidden_size, self.config["num_labels"])
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, labels, loss_weights=None, **kwargs):
        if "mt5" in self.config["pretrained_model_path"]:
            h = self.pretrained_model(
                decoder_input_ids=input_ids,
                attention_mask=attention_mask,
                input_ids=input_ids,
            )
        else:
            h = self.pretrained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # average pooling
        h = torch.mean(h.last_hidden_state, 1)

        h = self.hidden_layer(h)
        h = self.dropout(h)
        logits = self.classifier(h)

        loss = loss_fn(logits, labels, loss_weights)

        return logits, loss
