
from typing import List, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

DATE_COLUMNS = [
    "order_purchase_timestamp", "order_aproved_at",
    "order_estimated_delivery_date", "order_delivered_customer_date",
]


def read_data(dataset_path: str) -> pd.DataFrame:
   
    data = pd.read_csv(dataset_path)
    for column in DATE_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce")
    translation_path = Path(dataset_path).with_name("product_category_name_translation.csv")
    translations = pd.read_csv(translation_path)
    data = data.merge(translations, on="product_category_name", how="left",
                      sort=False, validate="many_to_one")
    return data.drop(columns="product_category_name")


def drop_unused_columns(
    data: pd.DataFrame, features_to_drop: Optional[List[str]]
) -> pd.DataFrame:
    
    return data.drop(columns=features_to_drop or [], errors="ignore")


def split_train_test_data(
    data: pd.DataFrame, target_col: str, test_size: float, random_state: int,
    groups: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
   
    if groups is not None:
        group_labels = pd.DataFrame({
            "group": groups.reindex(data.index), "target": data[target_col],
        }).groupby("group")["target"].agg(lambda values: values.mode().iloc[0])
        train_groups, test_groups = train_test_split(
            group_labels.index, test_size=test_size, random_state=random_state,
            stratify=group_labels,
        )
        return (data.loc[groups.isin(train_groups)], data.loc[groups.isin(test_groups)])
    train, test = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data[target_col]
    )
    return train, test
