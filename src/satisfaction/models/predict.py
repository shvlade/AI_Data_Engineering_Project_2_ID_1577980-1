
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

from satisfaction.models.fit import SklearnRegressionModel


def predict_model(model: SklearnRegressionModel, features: pd.DataFrame) -> np.ndarray:
    """Return continuous regression predictions without clipping ranking scores."""
    return model.predict(features)


def evaluate_model(
    predicts: np.ndarray, target: pd.Series, positive_threshold: int = 4
) -> Dict[str, float]:
    """Calculate RMSE and ROC-AUC for target >= 4; single-class AUC is NaN."""
    target = np.asarray(target).ravel()
    predicts = np.asarray(predicts).ravel()
    binary = target >= positive_threshold
    return {
        "rmse": float(np.sqrt(mean_squared_error(target, predicts))),
        "roc_auc": float(roc_auc_score(binary, predicts))
        if np.unique(binary).size == 2 else float("nan"),
    }
