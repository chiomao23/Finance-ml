"""
Tests for categorize.py.

Run with: pytest tests/
"""

import pytest

from categorize import predict_category, get_categories


def test_get_categories_returns_a_list():
    categories = get_categories()
    assert isinstance(categories, list)
    assert len(categories) > 0
    assert all(isinstance(c, str) for c in categories)


def test_predict_category_returns_expected_shape():
    result = predict_category("STARBUCKS STORE #221")
    assert "category" in result
    assert "confidence" in result
    assert isinstance(result["category"], str)
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predicted_category_is_a_known_category():
    result = predict_category("WHOLE FOODS MARKET")
    assert result["category"] in get_categories()


@pytest.mark.parametrize("text,expected_category", [
    ("STARBUCKS STORE #221", "Dining"),
    ("NETFLIX.COM", "Subscriptions"),
    ("WHOLE FOODS MARKET", "Groceries"),
    ("SHELL GAS STATION", "Transportation"),
])
def test_known_examples_predict_correctly(text, expected_category):
    """These are merchants the model was trained on, so we expect it to
    get them right. This catches accidental regressions if the model or
    preprocessing changes later."""
    result = predict_category(text)
    assert result["category"] == expected_category


@pytest.mark.parametrize("bad_input", ["", "   ", None, 123, []])
def test_invalid_input_raises_value_error(bad_input):
    with pytest.raises(ValueError):
        predict_category(bad_input)


def test_works_regardless_of_current_working_directory():
    """Regression test for the original path bug: this should work even
    if the caller's working directory isn't the categorization folder."""
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir("/tmp")
        result = predict_category("NETFLIX.COM")
        assert result["category"] == "Subscriptions"
    finally:
        os.chdir(original_cwd)
