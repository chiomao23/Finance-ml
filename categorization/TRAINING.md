# How categorizer.joblib was trained

This documents the actual steps used to produce the trained model, so it
can be reproduced or retrained if needed.

## 1. Data source

Dataset: [`DoDataThings/us-bank-transaction-categories-v2`](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2)
on Hugging Face — ~68,000 labeled bank transaction descriptions.

```python
from datasets import load_dataset
dataset = load_dataset("DoDataThings/us-bank-transaction-categories-v2")
df = dataset["train"].to_pandas()
```

## 2. Deduplication (important — do not skip)

The raw dataset contains a significant number of exact duplicate
descriptions (~33%, 22,303 of 68,000 rows). Without removing these first,
the same description can end up in both the training and test splits,
inflating the apparent accuracy. Deduplicate **before** splitting:

```python
df_unique = df.drop_duplicates(subset="description")
# 68,000 rows -> 45,697 unique rows
```

## 3. Train/test split

```python
from sklearn.model_selection import train_test_split

X = df_unique["description"]
y = df_unique["category"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## 4. Vectorize and train

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
```

## 5. Evaluate

```python
from sklearn.metrics import accuracy_score, classification_report

predictions = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
```

## 6. Save

```python
import joblib
joblib.dump({"vectorizer": vectorizer, "model": model}, "categorizer.joblib")
```

`random_state=42` is fixed throughout, so re-running these steps on the
same dataset reproduces the exact same model.
