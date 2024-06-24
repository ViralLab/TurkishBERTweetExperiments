import torch


class Dataset:
    def __init__(self, texts, target_labels, tokenizer, max_token_len: int = 256):
        self.texts = texts
        self.target_labels = target_labels
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        target_label = self.target_labels[item]
        ids = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_token_len,
        )

        return {
            "input_ids": torch.LongTensor(ids["input_ids"]),
            "attention_masks": torch.LongTensor(ids["attention_mask"]),
            "target_labels": torch.LongTensor([target_label]),
        }
