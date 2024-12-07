import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import adapters
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from huggingface_hub import login

login(token="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")

def load_and_preprocess_ag_news(tokenizer_name="bert-base-uncased", cache_dir="./dataset", max_length=140):
    # Load the ag_news dataset
    dataset = load_dataset("ag_news", cache_dir=cache_dir)

    # Split the dataset
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Preprocessing function
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    # Preprocess training and test datasets
    train_dataset = train_dataset.map(preprocess_function, num_proc=1, batched=True)
    test_dataset = test_dataset.map(preprocess_function, num_proc=1, batched=True)

    # Convert to PyTorch format
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return train_dataset, test_dataset

train_dataset, test_dataset = load_and_preprocess_ag_news(
    tokenizer_name="bert-base-uncased",
    cache_dir="./dataset",
    max_length=140
)

# Compute accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Experiment One: Fixed Adapter Size, Plot Test Set Benchmark Score Curve
def experiment_one():
    # Define different Adapter configurations
    adapter_configs = [
        ("seq_bn", "seq_bn"),
        ("lora", "lora"),
        ("double_seq_bn", "double_seq_bn"),
        ("seq_bn_inv", "seq_bn_inv"),
        ("double_seq_bn_inv", "double_seq_bn_inv"),
        ("compacter", "compacter"),
        ("compacter++", "compacter++"),
        ("prefix_tuning", "prefix_tuning"),
        ("prefix_tuning_flat", "prefix_tuning_flat"),
        ("ia3", "ia3"),
        ("mam", "mam"),
        ("unipelt", "unipelt"),
        ("prompt_tuning", "prompt_tuning"),
        ("loreft", "loreft"),
        ("noreft", "noreft"),
        ("direft", "direft"),
    ]

    # Store results
    adapter_sizes = []
    accuracies = []
    adapter_names = []
    all_epoch_accuracies = {}

    for name, config in adapter_configs:
        print(f"Running experiment with adapter: {name}, config: {config}")
        # Load BERT model
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

        # Initialize Adapter support
        adapters.init(model)

        try:
            # Add Adapter
            model.add_adapter(name, config=config)
            model.set_active_adapters(name)
            model.train_adapter(name)

            # Training parameters
            training_args = TrainingArguments(
                output_dir=f"./results_{name}",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=4e-4,
                per_device_train_batch_size=128,
                per_device_eval_batch_size=128,
                num_train_epochs=8,
                weight_decay=0.01,
                logging_steps=100,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                save_total_limit=1,
                bf16=True,
                dataloader_num_workers=8
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=compute_metrics,
            )

            # Train model
            trainer.train()

            # Get test set accuracy
            results = trainer.evaluate()
            accuracies.append(results["eval_accuracy"])

            # Get Adapter parameter count
            adapter_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            adapter_sizes.append(adapter_param_count)

            # Record Adapter name
            adapter_names.append(name)

            # Record accuracy for each epoch
            epoch_accuracies = [
                log["eval_accuracy"]
                for log in trainer.state.log_history
                if "eval_accuracy" in log
            ]
            all_epoch_accuracies[name] = epoch_accuracies

        except Exception as e:
            print(f"Error with adapter {name}: {e}")
            accuracies.append(0)  # If failed, record accuracy as 0
            adapter_sizes.append(0)  # Parameter count as 0
            adapter_names.append(name)
            all_epoch_accuracies[name] = [0] * 5  # If failed, record accuracy as 0

    # Plot Adapter parameter count vs. maximum accuracy
    plt.figure(figsize=(12, 8))
    bars = plt.bar(adapter_names, accuracies, color="skyblue")

    # Annotate accuracy values on the bar chart
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xlabel("Adapter Configurations")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy for Different Adapter Configurations")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("./figure/exp_1_max_accuracy.png")
    plt.show()

    # Plot accuracy changes over epochs for each Adapter
    plt.figure(figsize=(12, 8))
    for name, epoch_accuracies in all_epoch_accuracies.items():
        plt.plot(range(1, len(epoch_accuracies) + 1), epoch_accuracies, marker="o", label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epochs for Different Adapter Configurations")
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figure/exp_1_accuracy_curves.png")
    plt.show()

    # Print results
    for name, size, acc in zip(adapter_names, adapter_sizes, accuracies):
        print(f"Adapter: {name}, Parameters: {size}, Accuracy: {acc}")

def experiment_two():
    global tokenizer, train_dataset, test_dataset

    # Define different pretrained models
    model_names = [
        "prajjwal1/bert-small",
        "prajjwal1/bert-medium",
        "bert-base-uncased",
        "bert-large-uncased"
    ]

    # Store results
    model_sizes = []
    accuracies = []

    for model_name in model_names:
        print(f"Running experiment with model: {model_name}")
        # Load BERT model
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

        # Initialize Adapter support
        adapters.init(model)
        try:
            # Load model and tokenizer
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            train_dataset, test_dataset = load_and_preprocess_ag_news(
                tokenizer_name="FacebookAI/roberta-large",
                max_length=140
            )

            # Initialize Adapter support
            adapters.init(model)

            # Add Adapter
            adapter_name = "experiment2"
            model.add_adapter(adapter_name, config="lora")
            model.set_active_adapters(adapter_name)
            model.train_adapter(adapter_name)

            # Training parameters
            training_args = TrainingArguments(
                output_dir=f"./results_{model_name.replace('/', '_')}",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=4e-4,
                per_device_train_batch_size=128,
                per_device_eval_batch_size=64,
                num_train_epochs=5,
                weight_decay=0.01,
                logging_steps=100,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                save_total_limit=1,
                bf16=True,
                dataloader_num_workers=16
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=compute_metrics,
            )

            # Train model
            trainer.train()

            # Get test set accuracy
            results = trainer.evaluate()
            accuracies.append(results["eval_accuracy"])

            # Get model parameter count
            model_param_count = sum(p.numel() for p in model.parameters())
            model_sizes.append(model_param_count)

        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            accuracies.append(0)  # If failed, record accuracy as 0
            model_sizes.append(0)  # Parameter count as 0

    # Plot model vs. accuracy bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, color="lightgreen", alpha=0.8)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy for Different Pretrained Models")
    plt.xticks(rotation=45)

    # Annotate accuracy values on the bar chart
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.005, f"{acc:.4f}", ha="center", va="bottom", fontsize=10)

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("./figure/exp_4_accuracy.png")
    plt.show()

    # Print results
    for model_name, size, acc in zip(model_names, model_sizes, accuracies):
        print(f"Model: {model_name}, Parameters: {size}, Accuracy: {acc}")

def experiment_three():
    global tokenizer, train_dataset, test_dataset
    # Define different pretrained models
    model_names = [
        "google-bert/bert-base-uncased",
        "distilbert-base-uncased",
        "microsoft/deberta-base",
        "FacebookAI/roberta-base",
        "FacebookAI/roberta-large",
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
    ]

    # Set model cache directory
    os.environ["TRANSFORMERS_CACHE"] = "./model_cache"
    # Store results
    model_sizes = []
    accuracies = []

    for model_name in model_names:
        print(f"Running experiment with model: {model_name}")
        try:
            # Load model and tokenizer
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            train_dataset, test_dataset = load_and_preprocess_ag_news(
                tokenizer_name=model_name,
                max_length=140
            )
            # Initialize Adapter support
            adapters.init(model)

            # Add LoRA Adapter
            adapter_name = f"{model_name}_lora"
            config = adapters.LoRAConfig(r=32)
            model.add_adapter(adapter_name, config=config)
            model.set_active_adapters(adapter_name)
            model.train_adapter(adapter_name)

            # Preprocessing (based on current model's tokenizer)
            def preprocess_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=140)

            train_dataset_preprocessed = train_dataset.map(preprocess_function, batched=True)
            test_dataset_preprocessed = test_dataset.map(preprocess_function, batched=True)

            train_dataset_preprocessed.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
            test_dataset_preprocessed.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

            # Training parameters
            training_args = TrainingArguments(
                output_dir=f"./results_{model_name.replace('/', '_')}",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=4e-4,
                per_device_train_batch_size=64,
                per_device_eval_batch_size=64,
                num_train_epochs=5,
                weight_decay=0.01,
                logging_steps=100,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                save_total_limit=1,
                bf16=True,
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset_preprocessed,
                eval_dataset=test_dataset_preprocessed,
                compute_metrics=compute_metrics,
            )

            # Train model
            trainer.train()

            # Get test set accuracy
            results = trainer.evaluate()
            accuracies.append(results["eval_accuracy"])

            # Get model parameter count
            model_param_count = sum(p.numel() for p in model.parameters())
            model_sizes.append(model_param_count)

        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            accuracies.append(0)  # If failed, record accuracy as 0
            model_sizes.append(0)  # Parameter count as 0

    # Plot model vs. accuracy bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, color="lightcoral", alpha=0.8)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy for Different Pretrained Models")
    plt.xticks(rotation=45)

    # Annotate accuracy values on the bar chart
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.005, f"{acc:.4f}", ha="center", va="bottom", fontsize=10)

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("./figure/exp_5_accuracy.png")
    plt.show()

    # Print results
    for model_name, size, acc in zip(model_names, model_sizes, accuracies):
        print(f"Model: {model_name}, Parameters: {size}, Accuracy: {acc}")

def experiment_four():
    # Define different LoRA rank values
    lora_ranks = [2, 4, 8, 16, 32]

    # Store results
    lora_ranks_results = []
    accuracies = []

    for rank in lora_ranks:
        print(f"Running experiment with LoRA rank: {rank}")
        # Load BERT model
        model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-large", num_labels=4)
        train_dataset, test_dataset = load_and_preprocess_ag_news(
            tokenizer_name="FacebookAI/roberta-large",
            max_length=140
        )

        # Initialize Adapter support
        adapters.init(model)
        try:
            # Add LoRA Adapter
            adapter_name = f"lora_rank_{rank}"
            config = adapters.LoRAConfig(r=rank)
            model.add_adapter(adapter_name, config=config)
            model.set_active_adapters(adapter_name)
            model.train_adapter(adapter_name)

            # Training parameters
            training_args = TrainingArguments(
                output_dir=f"./results_{adapter_name}",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-4,
                per_device_train_batch_size=64,
                per_device_eval_batch_size=64,
                num_train_epochs=5,
                weight_decay=0.01,
                logging_steps=100,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                save_total_limit=1,
                bf16=True,
                dataloader_num_workers=16
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=compute_metrics,
            )

            # Train model
            trainer.train()

            # Get test set accuracy
            results = trainer.evaluate()
            accuracies.append(results["eval_accuracy"])

            # Get Adapter parameter count
            adapter_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            lora_ranks_results.append((rank, adapter_param_count))

        except Exception as e:
            print(f"Error with LoRA rank {rank}: {e}")
            accuracies.append(0)  # If failed, record accuracy as 0
            lora_ranks_results.append((rank, 0))  # Parameter count as 0

    # Plot LoRA rank vs. accuracy bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(lora_ranks, accuracies, color="skyblue", alpha=0.8, width=1.5)
    plt.xlabel("LoRA Rank", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Test Accuracy vs. LoRA Rank", fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Set x-axis ticks to lora_ranks
    plt.xticks(lora_ranks, labels=lora_ranks, fontsize=12)

    # Annotate accuracy values on the bar chart
    for i, acc in enumerate(accuracies):
        plt.text(lora_ranks[i], acc + 0.005, f"{acc:.4f}", ha="center", va="bottom", fontsize=10)

    # Save image and display
    plt.tight_layout()
    plt.savefig("./figure/exp_4_accuracy.png")
    plt.show()

    # Print results
    for rank, param_count, acc in zip(lora_ranks, [x[1] for x in lora_ranks_results], accuracies):
        print(f"LoRA Rank: {rank}, Parameters: {param_count}, Accuracy: {acc}")

def experiment_five():
    global tokenizer
    # Load BERT model
    model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-large", num_labels=4)
    train_dataset, test_dataset = load_and_preprocess_ag_news(
        tokenizer_name="FacebookAI/roberta-large",
        max_length=140
    )

    # Initialize Adapter support
    adapters.init(model)

    # Add Adapter
    config = adapters.LoRAConfig(r=32)
    model.add_adapter("experiment5", config=config)
    model.set_active_adapters("experiment5")

    model.train_adapter("experiment5")

    # Training parameters
    training_args = TrainingArguments(
        output_dir="./results_experiment5",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=64,
        num_train_epochs=15,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        bf16=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Train model and record results
    trainer.train()

    # Get test set accuracy
    results = trainer.evaluate()
    print("Final Test Accuracy (Experiment 5):", results["eval_accuracy"])

    # Plot test set accuracy curve
    epochs = list(range(0, training_args.num_train_epochs + 1))
    accuracies = [result["eval_accuracy"] for result in trainer.state.log_history if "eval_accuracy" in result]

    plt.plot(epochs, accuracies, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy vs. Epochs (Experiment 5)")
    plt.grid()
    plt.savefig("./figure/exp_5.png")
    plt.show()

def experiment_six():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")

    # Define 5 different classification task datasets
    classification_tasks = [
        "sms_spam",  # SMS spam classification
        "twitter_airline_sentiment",  # Airline sentiment classification
        "imdb",  # Movie review sentiment classification
        "trec",  # Question classification
        "ag_news",  # News classification
    ]

    # Store results for each task
    task_accuracies = {}

    for task in classification_tasks:
        print(f"Running experiment for task: {task}")

        # Load dataset
        if task == "sms_spam":
            dataset = load_dataset("sms_spam", cache_dir="./dataset", trust_remote_code=True)
            # SMS Spam dataset does not have a test set, need to split manually
            train_test_split = dataset["train"].train_test_split(test_size=0.2)
            dataset = DatasetDict({
                'train': train_test_split['train'],
                'test': train_test_split['test']
            })
        elif task == "twitter_airline_sentiment":
            dataset = load_dataset("tweet_eval", "sentiment", cache_dir="./dataset", trust_remote_code=True)
        elif task == "imdb":
            dataset = load_dataset("imdb", cache_dir="./dataset", trust_remote_code=True)
        else:
            dataset = load_dataset(task, cache_dir="./dataset", trust_remote_code=True)

        # Define preprocessing function based on task
        if task == "ag_news":
            # AG News dataset text field is "text"
            def preprocess_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

        elif task == "sms_spam":
            # SMS Spam dataset field is "sms"
            def preprocess_function(examples):
                return tokenizer(examples["sms"], padding="max_length", truncation=True, max_length=128)

        elif task == "twitter_airline_sentiment":
            # Twitter Airline Sentiment dataset field is "text"
            def preprocess_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

        elif task == "imdb":
            # IMDb dataset field is "text"
            def preprocess_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

        elif task == "trec":
            # TREC dataset field is "text"
            def preprocess_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

        else:
            raise ValueError(f"Unknown task: {task}")

        # Preprocess training and test datasets
        train_dataset = dataset["train"].map(preprocess_function, batched=True)
        test_dataset = dataset["test"].map(preprocess_function, batched=True)

        # Rename label column if necessary
        if "label" not in train_dataset.column_names:
            if "topic" in train_dataset.column_names:
                train_dataset = train_dataset.rename_column("topic", "label")
                test_dataset = test_dataset.rename_column("topic", "label")
            elif "fine_label" in train_dataset.column_names:
                train_dataset = train_dataset.rename_column("fine_label", "label")
                test_dataset = test_dataset.rename_column("fine_label", "label")

        # Convert to PyTorch format
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        # Load BERT model
        num_labels = int(max(train_dataset["label"]) + 1) if "label" in train_dataset.features else 2
        model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-large", num_labels=num_labels)

        # Initialize Adapter support
        adapters.init(model)

        # Add Adapter
        config = adapters.LoRAConfig(r=32)
        model.add_adapter(f"{task}_adapter", config=config)
        model.set_active_adapters(f"{task}_adapter")
        model.train_adapter(f"{task}_adapter")

        # Set training epochs for each task
        num_epoch_dict = {
            "ag_news": 5,
            "trec": 20,
            "sms_spam": 10,
            "twitter_airline_sentiment": 10,
            "imdb": 10,
        }

        # Training parameters
        training_args = TrainingArguments(
            output_dir=f"./results_{task}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-4,
            per_device_train_batch_size=128,
            per_device_eval_batch_size=32,
            num_train_epochs=num_epoch_dict[task],
            weight_decay=0.01,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,
            bf16=True
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        # Train model
        trainer.train()

        # Get test set accuracy
        results = trainer.evaluate()
        task_accuracies[task] = results["eval_accuracy"]

        print(f"Task: {task}, Final Test Accuracy: {results['eval_accuracy']}")

    # Print results for all tasks
    print("\nFinal Results for Experiment Six:")
    for task, accuracy in task_accuracies.items():
        print(f"Task: {task}, Accuracy: {accuracy}")

    # Plot task vs. accuracy bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(task_accuracies.keys(), task_accuracies.values(), color="lightblue", alpha=0.8)
    plt.xlabel("Task", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Test Accuracy for Different Classification Tasks", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=12)

    # Annotate accuracy values on the bar chart
    for i, acc in enumerate(task_accuracies.values()):
        plt.text(i, acc + 0.005, f"{acc:.4f}", ha="center", va="bottom", fontsize=10)

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("./figure/exp_6_accuracy.png")
    plt.show()

if __name__ == '__main__':
    # experiment_one()
    # experiment_two()
    # experiment_three()
    # experiment_four()
    # experiment_five()
    experiment_six()
