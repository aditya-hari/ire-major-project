import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# WaPo 
data = pd.read_csv("../../datasets/wapo.csv")

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

counts = np.bincount(np.array(train_labels))
most_common_class = np.argmax(counts)
predictions = np.array([most_common_class for i in range(len(test_labels))])
print("Wapo - ", f1_score(test_labels, predictions, average = 'weighted'))

# Enron
data = pd.read_csv("../../datasets/emails.csv")
data.rename(columns={"from":"author"}, inplace = True)

data["text"].replace('', np.nan, inplace = True)
data.dropna(inplace = True)

data["length"] = data["text"].apply(lambda x: len(x.split()))
data = data.drop(data[data.length > 500].index)
data = data.drop(data[data.length < 10].index)

items = data.author.value_counts().to_dict().items()
data = data[data.author.isin([key for key, val in items if val > 2000])]

texts = data.text.tolist()
labels = data.author.tolist()
label2id = {i: idx for (idx, i) in enumerate(sorted(set(labels)))}
id2label = {label2id[i]: i for i in label2id}
labels = [label2id[i] for i in labels]

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=.15, random_state = 42)

counts = np.bincount(np.array(train_labels))
most_common_class = np.argmax(counts)
predictions = np.array([most_common_class for i in range(len(test_labels))])
print("Enron - ", f1_score(test_labels, predictions, average = 'weighted'))

# Spooky Author 
data = pd.read_csv("../../datasets/spooky.csv")

data["text"].replace('', np.nan, inplace = True)
data.dropna(inplace = True)

texts = data.text.tolist()
labels = data.author.tolist()
label2id = {i: idx for (idx, i) in enumerate(sorted(set(labels)))}
id2label = {label2id[i]: i for i in label2id}
labels = [label2id[i] for i in labels]

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=.15, random_state = 42)
counts = np.bincount(np.array(train_labels))
most_common_class = np.argmax(counts)
predictions = np.array([most_common_class for i in range(len(test_labels))])
print("Spooky - ", f1_score(test_labels, predictions, average = 'weighted'))