from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth import apply_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from unsloth import standardize_sharegpt
import evaluate
from transformers import TextStreamer

# Configuration
max_seq_length = 512
batch_size = 128
gradient_accumulation_steps = 1
seed = 3407
output_dir = "outputs"
logging_steps = 10

# Load datasets
summary_train = load_dataset("json", data_files={"train": "./summary_train.jsonl"}, split="train")
summary_test = load_dataset("json", data_files={"test": "./summary_test.jsonl"}, split="test")
qa_train = load_dataset("json", data_files={"train": "./qa_train.jsonl"}, split="train")
qa_test = load_dataset("json", data_files={"test": "./qa_test.jsonl"}, split="test")
poetry_train = load_dataset("json", data_files={"train": "./poetry_train.jsonl"}, split="train")
poetry_test = load_dataset("json", data_files={"test": "./poetry_test.jsonl"}, split="test")

model_name = "unsloth/Llama-3.2-1B-Instruct"
_, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=False,
)

chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>"""
summary_train = standardize_sharegpt(summary_train)
# Convert dataset to ShareGPT format
summary_train = apply_chat_template(
    summary_train,
    tokenizer=tokenizer,
    chat_template=chat_template,
)
print(summary_train[0])

qa_train = standardize_sharegpt(qa_train)
qa_train = apply_chat_template(
    qa_train,
    tokenizer=tokenizer,
    chat_template=chat_template,
)

poetry_train = standardize_sharegpt(poetry_train)
poetry_train = apply_chat_template(
    poetry_train,
    tokenizer=tokenizer,
    chat_template=chat_template,
)

summary_test = standardize_sharegpt(summary_test)
summary_test = apply_chat_template(
    summary_test,
    tokenizer=tokenizer,
    chat_template=chat_template,
)

qa_test = standardize_sharegpt(qa_test)
qa_test = apply_chat_template(
    qa_test,
    tokenizer=tokenizer,
    chat_template=chat_template,
)

poetry_test = standardize_sharegpt(poetry_test)
poetry_test = apply_chat_template(
    poetry_test,
    tokenizer=tokenizer,
    chat_template=chat_template,
)

# Combine training and test datasets
train_datasets = {"summary": summary_train, "qa": qa_train, "poetry": poetry_train}
test_datasets = {"summary": summary_test, "qa": qa_test, "poetry": poetry_test}

# Define evaluation metrics
metric = evaluate.load("rouge")

# Fine-tuning and evaluation function
def train_and_evaluate(dataset_name, train_dataset, test_dataset):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False
    )

    # Add LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        max_seq_length=max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    # Configure training parameters
    training_args = TrainingArguments(
        do_eval=False,
        num_train_epochs=8,
        per_device_train_batch_size=batch_size,
        warmup_steps=10,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=logging_steps,
        output_dir=f"{output_dir}/{dataset_name}",
        optim="adamw_8bit",
        seed=seed,
        learning_rate=5e-4
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # Start training
    print(f"Training model on {dataset_name} dataset...")
    trainer.train()

    # Evaluate the model
    print(f"Evaluating model on {dataset_name} dataset...")

    FastLanguageModel.for_inference(model)

    # Inference phase
    predictions = []
    labels = []

    for example in test_dataset:
        message = [{"role": "user", "content": example["conversations"][0]['content']}]
        input_ids = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        # Generate output
        output = model.generate(input_ids, max_new_tokens=128,
                                pad_token_id=tokenizer.eos_token_id)
        predicted_text = str(tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True))
        predicted_text.replace("<|eot_id|>", "")
        predictions.append(predicted_text)
        labels.append(str(example["conversations"][1]['content']))

    model.save_pretrained(f"model/lora_{dataset_name}")
    tokenizer.save_pretrained(f"model/lora_{dataset_name}")

    # Calculate evaluation metrics
    results = metric.compute(predictions=predictions, references=labels)
    print(f"Results for {dataset_name}: {results['rougeL']}")

    return results

# Model name
# Fine-tune and evaluate on three datasets
results = {}
for dataset_name, train_dataset in train_datasets.items():
    test_dataset = test_datasets[dataset_name]
    results[dataset_name] = train_and_evaluate(dataset_name, train_dataset, test_dataset)

# Print final results
print("Final Results:")
for dataset_name, result in results.items():
    print(f"{dataset_name}: {result}")
