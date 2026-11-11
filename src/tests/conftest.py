
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_orders():
    count = 80
    scores = np.tile([1, 2, 3, 4, 5], count // 5)
    return pd.DataFrame({
        "order_id": [f"order-{i}" for i in range(count)],
        "review_score": scores,
        "review_comment_message": ["private review"] * count,
        "order_status": ["delivered"] * count,
        "customer_state": np.where(scores > 3, "SP", "RJ"),
        "product_category_name": ["casa"] * count,
        "order_products_value": np.arange(count) + 100.0,
        "order_freight_value": [10.0] * count,
        "order_items_qty": [2] * count,
        "order_sellers_qty": [1] * count,
        "product_name_lenght": [20] * count,
        "product_description_lenght": [100] * count,
        "product_photos_qty": [1] * count,
        "order_purchase_timestamp": ["2018-04-27 10:00:00"] * count,
        "order_aproved_at": ["2018-04-27 12:00:00"] * count,
        "order_estimated_delivery_date": ["2018-05-04"] * count,
        "order_delivered_customer_date": np.where(
            scores > 3, "2018-05-02 15:00:00", "2018-05-08 15:00:00"
        ),
    })


@pytest.fixture
def orders_path(tmp_path, raw_orders):
    path = tmp_path / "orders.csv"
    raw_orders.to_csv(path, index=False)
    pd.DataFrame({
        "product_category_name": ["casa"],
        "product_category_name_english": ["house"],
    }).to_csv(tmp_path / "product_category_name_translation.csv", index=False)
    return path
