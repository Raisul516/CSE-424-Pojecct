from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import transformers

# Print transformers version for debugging
print(f"Transformers version: {transformers.__version__}")

# Step 0: Check if CUDA is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# Step 1: Load Dataset
dataset = load_dataset("csv", data_files={"train": "soft_labeled_full_dataset.csv"})
print("Column names in the dataset:", dataset["train"].column_names)  # Debugging step

# Ensure the proper column names are used
if "post_text" in dataset["train"].column_names:
    dataset = dataset.rename_column("post_text", "text")  # Rename 'post_text' to 'text' for consistency

dataset = dataset["train"].train_test_split(test_size=0.2)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Print sample data to debug
print("Sample data point:", train_dataset[0])
print("Label type:", type(train_dataset[0]["label"]))

# Define label mapping
LABEL_MAP = {
    'none': 0,  # neutral
    'offensive': 1, 
    'hatespeech': 2
}

# Function to convert string labels to integers
def map_labels(examples):
    examples["label"] = [LABEL_MAP.get(str(label).lower(), 0) for label in examples["label"]]
    return examples

# Apply label mapping
train_dataset = train_dataset.map(map_labels, batched=True)
test_dataset = test_dataset.map(map_labels, batched=True)

# Step 2: Load Pre-Trained Model and Tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Move the model to GPU if available
model = model.to(device)

# Step 3: Tokenize Dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# Print sample after processing to verify format
print("Processed sample:", train_dataset[0])
print("Labels type:", type(train_dataset[0]["labels"]))

# Step 4: Define Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Step 5: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# Step 6: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Step 7: Train the Model
trainer.train()

# Step 8: Save the Fine-Tuned Model
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")

print("Model fine-tuning complete and saved in './fine_tuned_bert'.")

# Step 9: Generate Classification Report
model.eval()
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

# Convert numeric labels back to human-readable class names
label_names = {0: "normal", 1: "offensive", 2: "hatespeech"}
target_names = [label_names[i] for i in sorted(label_names.keys())]

# Print classification report
print("\n=== Classification Report ===")
print(classification_report(labels, preds, target_names=target_names))
print(f"Accuracy: {accuracy_score(labels, preds):.2%}")