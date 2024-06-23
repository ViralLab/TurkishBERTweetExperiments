from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_from_disk
import datasets as ds

ds.disable_caching()

model_id = "Orbina/Orbita-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]


def make_inference(example):
    messages = [
        {
            "role": "system",
            "content": "VRL-Orbita Türkçe metinlerin duygusunu verebilen bir chatbottur. (Pozitif, Nötr, Negatif)",
        },
        {"role": "user", "content": example["text"]},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    predicted_response = model.generate(
        input_ids,
        max_new_tokens=100,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    predicted_response = predicted_response[0][input_ids.shape[-1] :]
    predicted_output = tokenizer.decode(predicted_response, skip_special_tokens=True)
    example["predicted_output"] = predicted_output

    return example


ds_name = "BOUN"
dataset_path = f"DATAPATH/{ds_name}"
my_dataset = load_from_disk(dataset_path)
my_dataset = my_dataset.map(make_inference)

output_path = f"Orbita_output/{ds_name}"
# save the output
my_dataset.save_to_disk(output_path)
