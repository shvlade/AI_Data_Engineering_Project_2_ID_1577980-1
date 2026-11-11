
import pickle
from pathlib import Path
from typing import Union

import pandas as pd
from hydra.utils import instantiate
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

SklearnRegressionModel = Union[GradientBoostingRegressor, LinearRegression]


def train_model(
    model_params, train_features: pd.DataFrame, target: pd.Series
) -> SklearnRegressionModel:
    """Instantiate the configured regressor and fit on prepared features."""
    return instantiate(model_params).fit(train_features, target)


def serialize_model(model, output: str) -> str:
    """Persist a fitted estimator or transformer to a pickle file."""
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return output
