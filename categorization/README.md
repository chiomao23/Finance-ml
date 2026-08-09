# Transaction Categorization Model

Predicts a spending category (e.g. "Dining", "Groceries") for a bank
transaction description, along with the model's confidence.

## What this is

- **Approach:** TF-IDF (text → numbers) + Logistic Regression (scikit-learn)
- **Trained on:** ~68,000 labeled transaction descriptions
- **Not yet connected to the backend** — see "Integration status" below.

## Installation

```bash
pip install scikit-learn joblib
```

(If you're running this alongside the rest of the ML pipeline, use the
project's main `requirements.txt` instead.)

## Usage

```python
from categorize import predict_category, get_categories

# See every category the model can predict
print(get_categories())
# ['Dining', 'Entertainment', 'Fees', 'Groceries', 'Healthcare', 'Income',
#  'Rent', 'Shopping', 'Subscriptions', 'Transfer', 'Transportation', 'Utilities']

# Predict a category, with confidence
result = predict_category("STARBUCKS STORE #221")
print(result)
# {'category': 'Dining', 'confidence': 0.9326}
```

The model file (`categorizer.joblib`) is loaded automatically the first time
you import `categorize` — no setup call needed. It's located relative to
`categorize.py` itself, so this works no matter what folder you run your
program from (this was a real bug in an earlier version — see "Known
issues fixed" below).

### Handling invalid input

`predict_category()` raises `ValueError` for empty, whitespace-only, `None`,
or non-string input. Callers should catch this and handle it as a bad
request rather than letting it propagate:

```python
try:
    result = predict_category(transaction_description)
except ValueError as e:
    # e.g. return a 400 response if this is called from an API
    print(f"Invalid input: {e}")
```

## Categories the model predicts

```
Dining, Entertainment, Fees, Groceries, Healthcare, Income, Rent,
Shopping, Subscriptions, Transfer, Transportation, Utilities
```

(Run `get_categories()` for the authoritative, always-current list — it
reads directly from the trained model rather than being hardcoded here.)

## Model performance

Measured on a held-out test set (20% of the data, never seen during
training), **after fixing a data-leakage bug** where duplicate transaction
descriptions had appeared in both the training and test sets:

| Metric | Value |
|---|---|
| Accuracy | 99.44% |

Additionally tested against transactions written in a different style than
the training data (lowercase, abbreviated, reformatted — e.g. `"amzn mktp
us*a1b2c"` instead of `"AMAZON.COM*A1B2C3"`) to check real-world
generalization, not just performance on the same data style it trained on:

| Test | Result |
|---|---|
| 150 reformatted training-set examples | 149/150 correct (99.3%) |
| 20 hand-written, unfamiliar real-world transactions | 16/20 correct (80%) |

**Known weak spot:** informal phrasing for rent (e.g. "chk#1042 to
landlord") was misclassified as "Transfer" instead of "Rent" — the training
data's Rent examples used more formal phrasing (e.g. "RENT PAYMENT"). This
is a specific, understood gap, not a general reliability issue.

**Full precision/recall/F1 per category:** not yet regenerated on the
deduplicated data at time of writing. To produce this, re-run
`classification_report(y_test, predictions)` in the training notebook
against the corrected (deduplicated) train/test split, and paste the
output into this section.

## Tests

```bash
pytest tests/
```

Covers: correct output shape, valid category names, known-correct
predictions (regression protection), invalid-input handling, and that the
model loads correctly regardless of the caller's working directory.

## Known issues fixed

- **Path bug:** the model file was previously loaded using a path assumed
  relative to wherever the program happened to be run from. Fixed by
  resolving the path relative to `categorize.py`'s own location instead.
- **Data leakage:** ~33% of the original training dataset was duplicate
  rows, meaning some "test" examples had actually been seen during
  training. Fixed by deduplicating before splitting into train/test.

## Integration status

**Not yet wired up to the backend.** Per team discussion, the backend
integration approach (how the backend calls this service — direct import,
a REST endpoint, a message queue, etc.) is still being decided. This module
is designed to be integration-agnostic in the meantime: it's a plain
Python function with a clear input (a string) and output (a JSON-safe
dict), so it can be wrapped in whatever interface is agreed on later
without changing the underlying logic.
