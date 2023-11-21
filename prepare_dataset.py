import json

from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_prompt(example):
    question = example["input"].split("\n\n")[0]
    context = "".join(example["input"].split("\n\n")[1:])
    prompt = f"User: Context: {context} Question: {question}\n\nAssistant"
    return prompt


def filter_dataset(example, max_len=8192):
    prompt = prepare_prompt(example)
    if len(prompt.split()) > max_len:
        return False
    encoding = tokenizer(
        prompt,
        return_attention_mask=False,
        return_token_type_ids=False,
        padding=False,
        truncation=False,
    )
    return len(encoding["input_ids"]) < max_len


def save_to_jsonl(dataset, filename):
    lines = [{"input": prepare_prompt(x), "label": x["output"]} for x in dataset]
    with open(f"{filename}.jsonl", "w") as f:
        for line in lines:
            json_str = json.dumps(line)
            f.write(f"{json_str}\n")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    dataset = load_dataset("tau/scrolls", "contract_nli")
    train_dataset, val_dataset = dataset["train"], dataset["validation"]
    train_dataset = train_dataset.filter(filter_dataset, num_proc=4)
    val_dataset = val_dataset.filter(filter_dataset, num_proc=4)
    save_to_jsonl(train_dataset, "dataset/train")
    save_to_jsonl(val_dataset, "dataset/val")
