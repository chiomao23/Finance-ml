# Transaction Categorization Model

Predicts a spending category (e.g. "Dining", "Groceries") for a bank
transaction description, along with the model's confidence.

## What this is

- **Approach:** TF-IDF (text → numbers) + Logistic Regression (scikit-learn)
- **Trained on:** ~45,700 unique labeled transaction descriptions (deduplicated
  from an original 68,000 — roughly a third were exact duplicates, which
  would have caused the same text to appear in both training and test data)
- **Not yet connected to the backend** — see "Integration status" below.

## Installation

```bash
pip install scikit-learn joblib
```

## Usage

```python
from categorize import predict_category, get_categories

print(get_categories())
# ['Dining', 'Entertainment', 'Fees', 'Groceries', 'Healthcare', 'Income',
#  'Rent', 'Shopping', 'Subscriptions', 'Transfer', 'Transportation', 'Utilities']

result = predict_category("STARBUCKS STORE #221")
print(result)
# {'category': 'Dining', 'confidence': 0.9326}
```

### Handling invalid input

`predict_category()` raises `ValueError` for empty, whitespace-only, `None`,
or non-string input.

## Model performance
| Category | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Education | 1.00 | 1.00 | 1.00 | 419 |
| Entertainment | 1.00 | 1.00 | 1.00 | 499 |
| Fees | 1.00 | 1.00 | 1.00 | 270 |
| Groceries | 0.99 | 0.99 | 0.99 | 715 |
| Healthcare | 0.99 | 1.00 | 0.99 | 509 |
| Income | 0.99 | 1.00 | 0.99 | 448 |
| Insurance | 1.00 | 1.00 | 1.00 | 619 |
| Mortgage | 1.00 | 0.99 | 1.00 | 336 |
| Personal Care | 0.99 | 1.00 | 1.00 | 524 |
| Rent | 1.00 | 1.00 | 1.00 | 382 |
| Restaurants | 0.99 | 1.00 | 1.00 | 739 |
| Shopping | 0.99 | 0.97 | 0.98 | 793 |
| Subscription | 0.97 | 0.99 | 0.98 | 487 |
| Transfer | 0.99 | 0.99 | 0.99 | 574 |
| Transportation | 1.00 | 1.00 | 1.00 | 628 |
| Travel | 1.00 | 1.00 | 1.00 | 548 |
| Utilities | 1.00 | 1.00 | 1.00 | 650 |

**Overall accuracy:** 99% (9,140 test examples, deduplicated dataset)


**Known weak spot:** informal rent phrasing (e.g. "chk#1042 to landlord")
sometimes misclassified as "Transfer" — the training data's Rent examples
used more formal phrasing.

## Tests

```bash
pytest tests/
```

Covers: output shape, valid categories, known-correct predictions,
invalid-input handling, and that the model loads correctly regardless of
the caller's working directory.

## Known issues fixed

- **Path bug:** the model file is now located relative to `categorize.py`
  itself, not wherever the program happens to be run from.
- **Data leakage:** ~33% of the original dataset was duplicate rows,
  meaning some "test" examples had been seen during training. Fixed by
  deduplicating before splitting into train/test.

## Integration status

**Not yet wired up to the backend.** The interface (direct import, a REST
endpoint, etc.) is still being decided with the team.

