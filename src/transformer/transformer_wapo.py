import pandas as pd 
import numpy as np 
import torch
import time 

from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, AdamW, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

data = pd.read_csv("wapo.csv")

data["article"].replace('', np.nan, inplace = True)
data.dropna(inplace = True)

items = data.author.value_counts().to_dict().items()
data = data[data.author.isin([key for key, val in items if val > 99])]

texts = data.article.tolist()
labels = data.author.tolist()

label2id = {i: idx for (idx, i) in enumerate(sorted(set(labels)))}
id2label = {label2id[i]: i for i in label2id}

labels = [label2id[i] for i in labels]

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=.15, random_state = 42)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.15, random_state = 42)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased', max_length = 1024)

train_encodings = tokenizer(train_texts, truncation=True, padding='max_length')
test_encodings = tokenizer(test_texts, truncation=True, padding='max_length')
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

class WapoDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

train_dataset = WapoDataset(train_encodings, train_labels)
test_dataset = WapoDataset(test_encodings, test_labels)
val_dataset = WapoDataset(val_encodings, val_labels)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels = 25)

training_args = TrainingArguments(
    output_dir='./wapo_results',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./wapo_logs',            # directory for storing logs
    logging_steps=500,
    save_strategy = 'epoch',
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()