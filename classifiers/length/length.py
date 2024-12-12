import json
import logging
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from utils import JSONType

JOB_ID = os.getenv("SLURM_JOB_ID", random.randint(0, 2**31 - 1))
MAX_LENGTH = 512
NUM_BUCKETS = 8
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 5e-5
CLASSIFIER = "distilbert-base-uncased"
DATASET = Path("datasets/training_set.jsonl")
OUTPUT_DIRECTORY = Path(f"outputs/length-predictor-{JOB_ID}/")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("length-prediction")


class LlamaData(Dataset):
    def __init__(self, data: JSONType, tokenizer: DistilBertTokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        label = get_bucket(
            self.data[idx]["modelOutput"]["generation_token_count"],
            num_buckets=NUM_BUCKETS,
            max_length=MAX_LENGTH,
        )
        inputs = self.tokenizer(
            self.data[idx]["modelInput"]["prompt"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def get_bucket(token_count: int, num_buckets: int, max_length: int) -> int:
    """Allocates token lengths to buckets"""
    ranges = [int(max_length * (i + 1) / num_buckets) for i in range(num_buckets)]
    for i, upper_limit in enumerate(ranges):
        if token_count <= upper_limit:
            return i
    return num_buckets - 1

def compute_metrics(pred) -> JSONType:
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "f1": report["weighted avg"]["f1-score"],
    }


def plot_confusion_matrix(
        labels: np.ndarray, 
        preds: np.ndarray, 
        num_buckets: int, 
        max_length: int, 
        output_path: str,
    ) -> None:
    """Plots a confusion matrix for the results"""
    cm = confusion_matrix(labels, preds, labels=range(num_buckets))    
    cm_normalized = cm.astype('float') / cm.sum(axis=0, keepdims=True)

    # Define bucket ranges
    ranges = [int(max_length * (i + 1) / num_buckets) for i in range(num_buckets)]
    bucket_labels = [f"[{ranges[i-1]+1 if i > 0 else 1}, {ranges[i]}]" for i in range(num_buckets)]

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=bucket_labels)

    # Display the confusion matrix as normalized proportions
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)

    # Color bar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("Proportion of Actual Token Lengths in Each Predicted Bucket (Normalized)")

    # Show absolute numbers in the matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > 0.5 else "black")

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=bucket_labels,
           yticklabels=bucket_labels,
           ylabel="Actual Token Length",
           xlabel="Predicted Token Length")
    plt.title("Token Length Prediction: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_data(dataset_path: Path, model_name: str) -> JSONType:
    """Loads a JSONL dataset"""
    data = []
    with open(dataset_path, "r") as f:
        for line in f:
            j = json.loads(line)
            if model_name in j:
                data.append(j[model_name])
    return data


def main(dataset_path: Path, model_name: str, output_path: Path) -> None:
    """Fine-tunes DistilBERT to predict the response lengths of a model"""
    logger.info("Token Length Prediction")
    logger.info("dataset_path", dataset_path)
    logger.info("model_name", model_name)
    logger.info("output_path", output_path)

    # Load classifier
    classifier = DistilBertForSequenceClassification.from_pretrained(
        CLASSIFIER, 
        num_labels=NUM_BUCKETS,
    )
    tokenizer = DistilBertTokenizer.from_pretrained(CLASSIFIER)

    # Load dataset
    data = load_data(dataset_path, model_name)
    if not data:
        raise ValueError(f"No data found for model name: {model_name}")
    dataset = LlamaData(data, tokenizer)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # Train classifier
    training_args = TrainingArguments(
        output_dir=str(output_path),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,  # Keep only the best model
        logging_dir=str(output_path / "logs"),
        logging_steps=10,
        report_to="none",
    )
    trainer = Trainer(
        model=classifier,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # After training, trainer.model is the best model
    # Save the best model explicitly
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    # Evaluate classifier
    eval_results = trainer.evaluate(eval_dataset)
    logger.info(f"Best Eval Metrics: {eval_results}")

    # Get predictions
    predictions, labels, _ = trainer.predict(eval_dataset)
    preds = np.argmax(predictions, axis=1)
    labels = labels

    # Classification report
    logger.info(
        classification_report(
            labels,
            preds,
            target_names=[f"bucket_{i}" for i in range(NUM_BUCKETS)],
            zero_division=0,
        )
    )

    # Confusion matrix
    plot_confusion_matrix(
        labels,
        preds,
        num_buckets=NUM_BUCKETS,
        max_length=MAX_LENGTH,
        output_path=str(output_path / "confusion_matrix.png")
    )


if __name__ == "__main__":
    # Train for both Llama 3.2 Instruct (1B) and Llama 3.1 Instruct (3B)
    models = [
        {
            "model_name": "meta-llama/Llama-3.1-70B-Instruct",
            "output_path": OUTPUT_DIRECTORY / "70b",
        },
        {
            "model_name": "meta-llama/Llama-3.2-1B-Instruct",
            "output_path": OUTPUT_DIRECTORY / "1b",
        },
    ]

    # Train Models
    for config in models:
        os.makedirs(config["output_path"], exist_ok=True)
        logger.info(f"Training classifier: {config['model_name']} -> {config['output_path']}")
        main(DATASET, config["model_name"], config["output_path"])
