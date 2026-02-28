import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Load dataset
df = pd.read_csv("YoutubeCommentsDataSet.csv")
df = df.dropna().drop_duplicates()

# Encode labels (Positive / Neutral / Negative)
le = LabelEncoder()
df["Sentiment"] = le.fit_transform(df["Sentiment"])

# Save label mapping
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label Mapping:", label_mapping)

# Train-Test Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Comment"].tolist(),
    df["Sentiment"].tolist(),
    test_size=0.2,
    random_state=42
)

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenization
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Convert to Dataset
train_dataset = Dataset.from_dict({
    **train_encodings,
    "labels": train_labels
})

val_dataset = Dataset.from_dict({
    **val_encodings,
    "labels": val_labels
})

# Load Model (3 labels)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train
trainer.train()

# Save Model
model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")

print("✅ Model saved in sentiment_model folder")
