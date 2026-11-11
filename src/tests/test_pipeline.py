
import pickle
import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, ValidationError
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from satisfaction.data.make_dataset import read_data
from satisfaction.features.build_features import (
    TargetEncoder, build_transformer, engineer_features, make_features,
)
from satisfaction.models.fit import serialize_model, train_model
from satisfaction.models.predict import evaluate_model
from satisfaction.predict import run_inference
from satisfaction.train import prepare_val_features_for_predict, train_pipeline


@pytest.mark.parametrize("model", ["lr", "gbr"])
def test_config_is_typed(model):
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "configs"),
                               version_base=None):
        config = compose(config_name="config", overrides=[f"model={model}"])
        assert config.feature.target_col == "review_score"
        if model == "gbr":
            with pytest.raises(ValidationError):
                config.model.model_params.n_estimators = "not-an-integer"


def test_metrics():
    metrics = evaluate_model([1, 2, 4, 5], [1, 2, 4, 5])
    assert metrics == {"rmse": 0.0, "roc_auc": 1.0}
    assert np.isnan(evaluate_model([4, 5], [4, 5])["roc_auc"])
    assert evaluate_model([3.1, 3.2], [1, 5])["roc_auc"] == 1


def test_align_validation_columns():
    train = pd.DataFrame({"a": [1], "b": [2]})
    validation = pd.DataFrame({"b": [3], "extra": [4]}, index=[9])
    aligned = prepare_val_features_for_predict(train, validation)
    assert aligned.columns.tolist() == ["a", "b"]
    assert aligned.loc[9].tolist() == [0, 3]


@pytest.mark.parametrize("encoder", ["onehot", "target"])
def test_saved_inference_without_target(tmp_path, orders_path, encoder):
    data = engineer_features(read_data(str(orders_path)))
    transformer = build_transformer(["customer_state"], ["total_order_value"], encoder)
    transformer.fit(data, data.review_score)
    features = make_features(transformer, data)
    model = train_model({"_target_": "sklearn.linear_model.LinearRegression"},
                        features, data.review_score)
    model_path, transformer_path = tmp_path / "model.pkl", tmp_path / "transformer.pkl"
    serialize_model(model, str(model_path))
    serialize_model(transformer, str(transformer_path))
    raw = pd.read_csv(orders_path).drop(columns=["review_score", "review_comment_message"])
    raw.to_csv(orders_path, index=False)
    output = run_inference(str(model_path), str(transformer_path), str(orders_path),
                           str(tmp_path / "result.csv"))
    assert output.columns.tolist() == ["order_id", "predicted_review_score"]
    assert len(output) == len(raw)
    np.testing.assert_allclose(output.predicted_review_score,
                               np.clip(model.predict(features), 1, 5))
    with open(transformer_path, "rb") as stream:
        restored = pickle.load(stream)
    np.testing.assert_allclose(make_features(restored, data), features)


class EncodingRegressor(RegressorMixin, BaseEstimator):
    def fit(self, x, y):
        return self

    def predict(self, x):
        return np.asarray(x).ravel()


def test_target_encoding_fits_inside_cv():
    x = pd.DataFrame({"category": list("abcdefgh")})
    y = np.arange(8, dtype=float)
    pipeline = Pipeline([("encode", TargetEncoder(0)), ("model", EncodingRegressor())])
    splits = list(KFold(4).split(x))
    scores = cross_val_score(pipeline, x, y, cv=splits, scoring="neg_mean_squared_error")
    expected = [-np.mean((y[test] - y[train].mean()) ** 2) for train, test in splits]
    np.testing.assert_allclose(scores, expected)


@pytest.mark.parametrize("encoder", ["onehot", "target"])
def test_train_pipeline_mlflow(tmp_path, orders_path, monkeypatch, encoder):
    import mlflow
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "configs"),
                               version_base=None):
        config = compose(config_name="config", overrides=[
            "model=lr", f"feature.categorical_encoder={encoder}",
        ])
        config.dataset.input_data_path = str(orders_path)
        params = OmegaConf.to_object(config)
    tracking = tmp_path / "tracking"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking.as_uri())
    monkeypatch.setenv("MLFLOW_EXPERIMENT", "integration")
    monkeypatch.chdir(tmp_path)
    train_pipeline(params)
    assert (tmp_path / "models" / "lin-reg.pkl").is_file()
    assert (tmp_path / "models" / "transformer.pkl").is_file()
    runs = mlflow.search_runs(experiment_names=["integration"])
    assert len(runs) == 1 and runs.iloc[0]["status"] == "FINISHED"
    assert runs.iloc[0]["tags.mlflow.source.name"] == "src/satisfaction/train.py"
    assert runs.iloc[0]["metrics.roc_auc"] > 0.65
    client = mlflow.MlflowClient()
    artifacts = client.list_artifacts(runs.iloc[0].run_id, "models")
    assert {a.path for a in artifacts} == {"models/lin-reg.pkl", "models/transformer.pkl"}
