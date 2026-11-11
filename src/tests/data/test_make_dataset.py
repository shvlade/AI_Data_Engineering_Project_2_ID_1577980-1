
import numpy as np
import pandas as pd
import pytest
from satisfaction.data.make_dataset import (
    read_data, drop_unused_columns, split_train_test_data,
)


def test_split_dataset():
    data = pd.DataFrame({"review_score": np.repeat([1, 2, 3, 4, 5], [100, 50, 100, 200, 550])})
    train, test = split_train_test_data(data, "review_score", 0.25, 42)
    assert len(train) == 750 and len(test) == 250
    assert set(train.index).isdisjoint(test.index)
    assert set(train.index) | set(test.index) == set(data.index)
    expected = data.review_score.value_counts(normalize=True).sort_index()
    for sample in (train, test):
        actual = sample.review_score.value_counts(normalize=True).sort_index()
        np.testing.assert_allclose(actual, expected, atol=0.005)
    second = split_train_test_data(data, "review_score", 0.25, 42)
    pd.testing.assert_frame_equal(train, second[0])


def test_read_data_merge(orders_path):
    data = read_data(str(orders_path))
    assert len(data) == 80
    assert data.product_category_name_english.eq("house").all()
    assert "product_category_name" not in data
    assert pd.api.types.is_datetime64_any_dtype(data.order_aproved_at)
    assert data.order_id.tolist() == [f"order-{i}" for i in range(80)]


def test_translation_duplicate_rejected(orders_path):
    lookup = orders_path.with_name("product_category_name_translation.csv")
    data = pd.read_csv(lookup)
    pd.concat([data, data]).to_csv(lookup, index=False)
    with pytest.raises(pd.errors.MergeError):
        read_data(str(orders_path))


def test_unmatched_category_preserves_row(orders_path):
    data = pd.read_csv(orders_path)
    data.loc[0, "product_category_name"] = "unknown"
    data.to_csv(orders_path, index=False)
    merged = read_data(str(orders_path))
    assert len(merged) == len(data)
    assert pd.isna(merged.loc[0, "product_category_name_english"])


def test_drop_unused_columns(raw_orders):
    result = drop_unused_columns(raw_orders, ["review_comment_message", "order_id", "absent"])
    assert "order_id" not in result and "review_comment_message" not in result
    assert "order_id" in raw_orders
    pd.testing.assert_frame_equal(drop_unused_columns(raw_orders, None), raw_orders)


def test_group_split_keeps_orders_together():
    data = pd.DataFrame({
        "order_id": np.repeat(np.arange(100), 2),
        "review_score": np.tile([1, 2, 3, 4, 5], 40),
    })
    train, test = split_train_test_data(data, "review_score", 0.25, 42, data.order_id)
    assert len(train) + len(test) == len(data)
    assert len(test) == 50
    assert set(train.order_id).isdisjoint(test.order_id)
    labels = data.groupby("order_id").review_score.agg(lambda values: values.mode().iloc[0])
    expected = labels.value_counts(normalize=True).sort_index()
    actual = labels.loc[test.order_id.unique()].value_counts(normalize=True).sort_index()
    np.testing.assert_allclose(actual, expected, atol=0.04)
