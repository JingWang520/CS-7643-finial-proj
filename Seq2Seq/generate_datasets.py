from datasets import load_dataset
from transformers import AutoTokenizer
import json
import random

from huggingface_hub import login

# Log in to Hugging Face
login(token="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")

# Load the tokenizer for Llama3 1B
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Set the maximum token length
MAX_TOKEN_LENGTH = 512

# Define a filter function (supports batch processing)
def filter_by_length_batch(batch):
    inputs = batch["input"]
    outputs = batch["output"]

    # Use the tokenizer's batch processing feature with multithreading enabled
    input_encodings = tokenizer(
        inputs,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
        padding=False,
        return_length=True,
    )
    output_encodings = tokenizer(
        outputs,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
        padding=False,
        return_length=True,
    )

    # Get the length of each input and output
    input_lengths = input_encodings["length"]
    output_lengths = output_encodings["length"]

    # Calculate the total length and return the filter result
    return [input_len + output_len <= MAX_TOKEN_LENGTH for input_len, output_len in zip(input_lengths, output_lengths)]

# Define a function to save data as JSONL
def save_to_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

# Define a function to convert data to the target format
def convert_to_conversations(data):
    formatted_data = []
    for item in data:
        conversation = {
            "conversations": [
                {"content": item["input"], "from": "user"},
                {"content": item["output"], "from": "assistant"}
            ]
        }
        formatted_data.append(conversation)
    return formatted_data

# Modify the function to save as JSONL to support the target format
def save_conversations_to_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

if __name__ == '__main__':
    # Load the summarization dataset
    summary_dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
    summary_data = summary_dataset.map(
        lambda x: {
            "instruction": "",
            "input": f"Summarize the following article:\n\n{x['article']}",
            "output": x["highlights"],
            "text": f"Summarize the following article:\n\n{x['article']}\n{x['highlights']}"
        },
    )

    summary_data = summary_data.select_columns(['instruction', 'input', 'output', "text"])
    print("filter summary_data")
    summary_data = summary_data.filter(lambda x: len(x["input"]) + len(x["output"]) < 1500, num_proc=12)
    summary_data = summary_data.filter(filter_by_length_batch, batched=True, batch_size=100).select(range(4000))

    # Load the QA dataset
    qa_dataset = load_dataset("squad", split="train")
    qa_data = qa_dataset.map(
        lambda x: {
            "instruction": "",
            "input": f"context:\n{x['context']}\n\nquestion:{x['question']}",
            "output": x["answers"]["text"][0],
            "text": f"{x['question']}\n{x['answers']['text'][0]}"
        },
        num_proc=4
    )
    print("filter qa_data")
    qa_data = qa_data.select_columns(['instruction', 'input', 'output', "text"])

    qa_data = qa_data.filter(lambda x: len(x["input"]) + len(x["output"]) < 2048, num_proc=12)
    qa_data = qa_data.filter(filter_by_length_batch, num_proc=16, batched=True, batch_size=100).select(range(4000))

    # Load the poetry dataset (Poetry Foundation)
    poetry_dataset = load_dataset("merve/poetry", split="train")
    poetry_data = poetry_dataset.map(
        lambda x: {
            "instruction": "",
            "input": f"Write a poem based on the following title:\n\n{x['poem name']}",
            "output": x["content"],
            "text": f"Write a poem based on the following title:\n\n{x['poem name']}\n{x['content']}"
        },
    )
    poetry_data = poetry_data.select_columns(['instruction', 'input', 'output', "text"])

    print("filter poetry_data")
    poetry_data = poetry_data.filter(lambda x: len(x["input"]) + len(x["output"]) < 2048, num_proc=12)
    poetry_data = poetry_data.filter(filter_by_length_batch, num_proc=16, batched=True, batch_size=100)

    # Split the data into training and test sets
    summary_data = summary_data.shuffle(seed=42)
    qa_data = qa_data.shuffle(seed=42)
    poetry_data = poetry_data.shuffle(seed=42)

    summary_train = summary_data.select(range(int(len(summary_data) * 0.9)))
    summary_test = summary_data.select(range(int(len(summary_data) * 0.9), len(summary_data)))

    qa_train = qa_data.select(range(int(len(qa_data) * 0.9)))
    qa_test = qa_data.select(range(int(len(qa_data) * 0.9), len(qa_data)))

    poetry_train = poetry_data.select(range(int(len(poetry_data) * 0.9)))
    poetry_test = poetry_data.select(range(int(len(poetry_data) * 0.9), len(poetry_data)))

    # Convert to the target format
    summary_train_conversations = convert_to_conversations(summary_train)
    summary_test_conversations = convert_to_conversations(summary_test)

    qa_train_conversations = convert_to_conversations(qa_train)
    qa_test_conversations = convert_to_conversations(qa_test)

    poetry_train_conversations = convert_to_conversations(poetry_train)
    poetry_test_conversations = convert_to_conversations(poetry_test)

    # Save as JSONL files
    save_conversations_to_jsonl(summary_train_conversations, "summary_train.jsonl")
    save_conversations_to_jsonl(summary_test_conversations, "summary_test.jsonl")

    save_conversations_to_jsonl(qa_train_conversations, "qa_train.jsonl")
    save_conversations_to_jsonl(qa_test_conversations, "qa_test.jsonl")

    save_conversations_to_jsonl(poetry_train_conversations, "poetry_train.jsonl")
    save_conversations_to_jsonl(poetry_test_conversations, "poetry_test.jsonl")
