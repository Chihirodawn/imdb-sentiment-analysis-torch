import logging
import os
import sys

import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


MODEL_NAME = "microsoft/deberta-v3-large"
RESULT_PATH = "./result/deberta_v3_large_trainer.csv"
OUTPUT_DIR = "./checkpoint/deberta_v3_large_trainer"
LOGGING_DIR = "./logs/deberta_v3_large_trainer"
MAX_LENGTH = 512
RANDOM_SEED = 42


def main():
    program = os.path.basename(sys.argv[0])
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    logging.root.setLevel(level=logging.INFO)
    logging.getLogger(program).info("running %s", " ".join(sys.argv))

    set_seed(RANDOM_SEED)
    train = pd.read_csv(
        "./corpus/imdb/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3
    )
    test = pd.read_csv(
        "./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3
    )
    train, val = train_test_split(
        train,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=train["sentiment"],
    )

    train_dataset = datasets.Dataset.from_dict(
        {"label": train["sentiment"], "text": train["review"]}
    )
    val_dataset = datasets.Dataset.from_dict(
        {"label": val["sentiment"], "text": val["review"]}
    )
    test_dataset = datasets.Dataset.from_dict({"text": test["review"]})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def preprocess_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=MAX_LENGTH
        )

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": float((predictions == labels).mean())}

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=LOGGING_DIR,
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch",
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs.predictions, axis=-1).flatten()
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    pd.DataFrame({"id": test["id"], "sentiment": test_pred}).to_csv(
        RESULT_PATH, index=False, quoting=3
    )
    logging.info("result saved to %s", RESULT_PATH)


if __name__ == "__main__":
    main()
