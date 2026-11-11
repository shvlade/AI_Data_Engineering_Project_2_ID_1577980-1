
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from satisfaction.features.build_features import (
    AttributesAdder, OutlierRemover, build_numerical_pipeline, build_transformer,
)


def test_attributes_adder(raw_orders):
    original = raw_orders.copy(deep=True)
    result = AttributesAdder().transform(raw_orders)
    assert result.loc[0, "wd_estimated_delivery_time"] == 4
    assert result.loc[0, "wd_actual_delivery_time"] == 6
    assert result.loc[0, "wd_delivery_time_delta"] == 2
    assert result.loc[0, "is_late"] == 1
    assert result.loc[0, "average_product_value"] == 50
    assert result.loc[0, "total_order_value"] == 110
    assert result.loc[0, "order_freight_ratio"] == 0.1
    assert result.loc[0, "purchase_dayofweek"] == 4
    assert "order_aproved_at" not in result
    pd.testing.assert_frame_equal(raw_orders, original)


def test_missing_dates_and_zero_denominators(raw_orders):
    raw_orders.loc[0, "order_delivered_customer_date"] = None
    raw_orders.loc[0, ["order_items_qty", "order_products_value"]] = 0
    output = AttributesAdder().transform(raw_orders)
    assert output.loc[0, ["is_late", "wd_actual_delivery_time", "average_product_value",
                          "order_freight_ratio"]].isna().all()
    assert not np.isinf(output.select_dtypes("number").to_numpy()).any()


def test_outlier_bounds_do_not_depend_on_inference_batch():
    remover = OutlierRemover().fit(pd.DataFrame({"value": range(10)}))
    single = pd.DataFrame({"value": [20.0]})
    batch = pd.DataFrame({"value": [20.0, 1e8, 1e9]})
    assert pd.isna(remover.transform(single).iloc[0, 0])
    assert pd.isna(remover.transform(batch).iloc[0, 0])
    assert remover.upper_.iloc[0] == 13.5


def test_zero_iqr_preserves_binary_signal():
    data = pd.DataFrame({"is_late": [0] * 99 + [1]})
    pd.testing.assert_frame_equal(OutlierRemover().fit_transform(data), data)


def test_outlier_requires_fit():
    with pytest.raises(NotFittedError):
        OutlierRemover().transform([[1]])


def test_empty_numeric_column_and_infinity():
    data = pd.DataFrame({"empty": [np.nan] * 5, "x": [1, 2, np.inf, 3, 4]})
    result = build_numerical_pipeline().fit_transform(data)
    assert result.shape == (5, 2)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("encoder", ["onehot", "target"])
def test_column_transformer_excludes_target_and_ids(encoder):
    data = pd.DataFrame({"cat": ["a", "b", "a", "b"], "num": [1, 2, 3, 4],
                         "review_score": [1, 2, 4, 5], "order_id": ["a", "b", "c", "d"]})
    transformer = build_transformer(["cat"], ["num"], encoder).fit(data, data.review_score)
    expected = transformer.transform(data)
    changed = data.assign(review_score=999, order_id="changed")
    actual = transformer.transform(changed)
    np.testing.assert_allclose(actual, expected)
