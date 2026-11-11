
from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from workalendar.america import Brazil

from satisfaction.data.make_dataset import DATE_COLUMNS

BR_CAL = Brazil()


@lru_cache(maxsize=100_000)
def _working_days(start, end) -> float:
    
    if pd.isna(start) or pd.isna(end):
        return np.nan
    if end < start:
        return -float(BR_CAL.get_working_days_delta(end, start))
    return float(BR_CAL.get_working_days_delta(start, end))


class AttributesAdder(BaseEstimator, TransformerMixin):
   

    def fit(self, x: pd.DataFrame, y=None):
        
        return self

    def transform(self, x: pd.DataFrame, y=None) -> pd.DataFrame:
        
        data = x.copy()
        for column in DATE_COLUMNS:
            data[column] = pd.to_datetime(data[column], errors="coerce")
        approved = data["order_aproved_at"].dt.date
        for suffix, column in (
            ("estimated", "order_estimated_delivery_date"),
            ("actual", "order_delivered_customer_date"),
        ):
            data[f"wd_{suffix}_delivery_time"] = [
                _working_days(start, end)
                for start, end in zip(approved, data[column].dt.date)
            ]
        data["wd_delivery_time_delta"] = (
            data["wd_actual_delivery_time"] - data["wd_estimated_delivery_time"]
        )
        actual = data["order_delivered_customer_date"]
        expected = data["order_estimated_delivery_date"]
        data["is_late"] = (actual > expected).astype(float).where(
            actual.notna() & expected.notna()
        )
        value = data["order_products_value"]
        data["average_product_value"] = value / data["order_items_qty"].replace(0, np.nan)
        data["total_order_value"] = value + data["order_freight_value"]
        data["order_freight_ratio"] = data["order_freight_value"] / value.replace(0, np.nan)
        data["purchase_dayofweek"] = data["order_purchase_timestamp"].dt.dayofweek
        return data.drop(columns=DATE_COLUMNS)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    
    return AttributesAdder().transform(df)


class TargetEncoder(BaseEstimator, TransformerMixin):
    

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing

    def fit(self, x, y):
        
        if self.smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        data = pd.DataFrame(x)
        target = np.asarray(y, dtype=float).ravel()
        if len(data) != len(target) or not np.isfinite(target).all() or target.size == 0:
            raise ValueError("y must contain one finite target per training row")
        self.n_features_in_ = data.shape[1]
        self.global_mean_ = float(target.mean())
        self.mappings_ = {}
        for column in data.columns:
            grouped = pd.DataFrame({"category": data[column].to_numpy(), "target": target})
            stats = grouped.groupby("category")["target"].agg(["sum", "count"])
            self.mappings_[column] = (
                (stats["sum"] + self.global_mean_ * self.smoothing)
                / (stats["count"] + self.smoothing)
            ).to_dict()
        self.columns_ = list(data.columns)
        return self

    def transform(self, x) -> pd.DataFrame:
        
        check_is_fitted(self, "mappings_")
        data = pd.DataFrame(x).copy()
        if data.shape[1] != self.n_features_in_:
            raise ValueError("TargetEncoder feature count differs from training")
        if isinstance(x, pd.DataFrame) and list(data.columns) != self.columns_:
            raise ValueError("TargetEncoder columns differ from training")
        for position, column in enumerate(self.columns_):
            data[data.columns[position]] = (
                data.iloc[:, position].map(self.mappings_[column]).fillna(self.global_mean_)
            )
        return data.astype(float)


def build_categorical_pipeline(encoder: str = "onehot") -> Pipeline:
    
        categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    elif encoder == "target":
        categorical_encoder = TargetEncoder()
    else:
        raise ValueError(f"Unknown categorical encoder {encoder!r}; use onehot or target")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", categorical_encoder),
    ])


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    
    transformed = build_categorical_pipeline().fit_transform(categorical_df)
    return pd.DataFrame(transformed.toarray() if hasattr(transformed, "toarray")
                        else transformed, index=categorical_df.index)


class OutlierRemover(BaseEstimator, TransformerMixin):
   

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, x, y=None):
        
        if self.factor < 0:
            raise ValueError("factor must be non-negative")
        data = pd.DataFrame(x).replace([np.inf, -np.inf], np.nan)
        self.n_features_in_ = data.shape[1]
        self.columns_ = list(data.columns)
        first, third = data.quantile(0.25), data.quantile(0.75)
        spread = third - first
        self.lower_ = (first - self.factor * spread).where(spread > 0, -np.inf)
        self.upper_ = (third + self.factor * spread).where(spread > 0, np.inf)
        return self

    def transform(self, x) -> pd.DataFrame:
       
        check_is_fitted(self, "lower_")
        data = pd.DataFrame(x).copy().replace([np.inf, -np.inf], np.nan)
        if data.shape[1] != self.n_features_in_ or list(data.columns) != self.columns_:
            raise ValueError("OutlierRemover columns differ from training")
        return data.mask(data.lt(self.lower_) | data.gt(self.upper_))

    def outlier_removal(self, x: pd.Series) -> pd.Series:
        
        return OutlierRemover(self.factor).fit_transform(x.to_frame()).iloc[:, 0]


def build_numerical_pipeline() -> Pipeline:
    
        ("outliers", OutlierRemover()),
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler(with_mean=False)),
    ])


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
   
    return pd.DataFrame(build_numerical_pipeline().fit_transform(numerical_df),
                        index=numerical_df.index)


def build_transformer(
    categorical_features: List[str], numerical_features: List[str],
    categorical_encoder: str = "onehot",
) -> ColumnTransformer:
   
    return ColumnTransformer([
        ("categorical_pipeline", build_categorical_pipeline(categorical_encoder),
         list(categorical_features)),
        ("numerical_pipeline", build_numerical_pipeline(), list(numerical_features)),
    ])


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    
    transformed = transformer.transform(df)
    return pd.DataFrame(transformed.toarray() if hasattr(transformed, "toarray")
                        else transformed, index=df.index)


def extract_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    
    return df[target_col].copy()
