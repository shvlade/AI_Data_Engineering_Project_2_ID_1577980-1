
import hashlib
import json
import logging
import os
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

from satisfaction.data.make_dataset import (
    drop_unused_columns, read_data, split_train_test_data,
)
from satisfaction.entities import TrainingPipelineConfig
from satisfaction.features.build_features import (
    build_transformer, engineer_features, extract_target, make_features,
)
from satisfaction.models.fit import serialize_model, train_model
from satisfaction.models.predict import evaluate_model, predict_model

logger = logging.getLogger(__name__)


def prepare_val_features_for_predict(
    train_features: pd.DataFrame, val_features: pd.DataFrame
) -> pd.DataFrame:
    return train_features.align(val_features, join="left", axis=1)[1].fillna(0)


def regression_roc_auc(estimator, features, target) -> float:

    return evaluate_model(estimator.predict(features), target)["roc_auc"]


def cross_validate_model(params, transformer, features, target, groups=None) -> dict:
    pipeline = Pipeline([
        ("transformer", transformer), ("model", instantiate(params.model.model_params)),
    ])
    splitter = StratifiedKFold if groups is None else StratifiedGroupKFold
    folds = splitter(n_splits=3, shuffle=True, random_state=params.split.random_state)
    scores = cross_val_score(
        pipeline, features, target, cv=folds.split(features, target, groups),
        scoring=regression_roc_auc, n_jobs=1, error_score="raise",
    )
    logger.info("CV ROC-AUC: folds=%s, mean=%.6f, std=%.6f",
                scores.tolist(), scores.mean(), scores.std())
    return {"cv_roc_auc_mean": float(scores.mean()),
            "cv_roc_auc_std": float(scores.std())}


def _run_directory() -> Path:

    if HydraConfig.initialized():
        return Path(HydraConfig.get().runtime.output_dir)
    return Path.cwd()


def _save_artifacts(params, model, transformer, metrics: dict, manifest: dict) -> None:
    output = _run_directory()
    model_path = output / "models" / f"{params.model.model_name}.pkl"
    transformer_path = output / "models" / "transformer.pkl"
    metrics_path = output / "metrics" / f"{params.model.model_name}.json"
    serialize_model(model, str(model_path))
    serialize_model(transformer, str(transformer_path))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    mlflow.log_artifact(str(model_path), artifact_path="models")
    mlflow.log_artifact(str(transformer_path), artifact_path="models")
    mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
    logger.info("Saved model, transformer and metrics in %s", output)


def train_pipeline(params: TrainingPipelineConfig) -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT", "satisfaction")
    if mlflow.get_tracking_uri().startswith(("http://", "https://")):
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name, artifact_location="mlflow-artifacts:/")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(
        run_name=params.model.model_name,
        tags={"mlflow.source.name": "src/satisfaction/train.py"},
    ) as run:
        model_config = OmegaConf.structured(params.model.model_params)
        mlflow.log_params(OmegaConf.to_container(model_config, resolve=True))
        mlflow.log_params({
            "split.test_size": params.split.test_size,
            "split.random_state": params.split.random_state,
            "categorical_encoder": params.feature.categorical_encoder,
        })
        dataset_path = Path(to_absolute_path(params.dataset.input_data_path))
        data = read_data(str(dataset_path))
        logger.info("Read dataset: shape=%s", data.shape)
        if data["order_id"].isna().any():
            raise ValueError("order_id must not be missing")
        order_groups = data["order_id"].copy()
        logger.info("Unique orders=%d; extra rows=%d; splitting by order_id",
                    order_groups.nunique(), order_groups.duplicated().sum())
        if not data[params.feature.target_col].isin([1, 2, 3, 4, 5]).all():
            raise ValueError("review_score must contain only 1, 2, 3, 4, 5")
        data = drop_unused_columns(data, params.feature.features_to_drop)
        data = engineer_features(data)
        logger.info("Engineered dataset: shape=%s", data.shape)
        train_df, test_df = split_train_test_data(
            data, params.feature.target_col, params.split.test_size, params.split.random_state,
            groups=order_groups,
        )
        logger.info("Stratified split: train=%d test=%d", len(train_df), len(test_df))
        train_target = extract_target(train_df, params.feature.target_col)
        test_target = extract_target(test_df, params.feature.target_col)
        train_df = train_df.drop(columns=params.feature.target_col)
        test_df = test_df.drop(columns=params.feature.target_col)
        transformer = build_transformer(
            params.feature.categorical_features, params.feature.numerical_features,
            params.feature.categorical_encoder,
        )
        metrics = cross_validate_model(
            params, transformer, train_df, train_target, order_groups.loc[train_df.index]
        )
        mlflow.log_metrics(metrics)
        logger.info("Fitting final preprocessor and %s", params.model.model_name)
        transformer.fit(train_df, train_target)
        train_features = make_features(transformer, train_df)
        test_features = prepare_val_features_for_predict(
            train_features, make_features(transformer, test_df)
        )
        model = train_model(params.model.model_params, train_features, train_target)
        metrics.update(evaluate_model(predict_model(model, test_features), test_target))
        mlflow.log_metrics(metrics)
        logger.info("Final test metrics: %s", metrics)
        manifest = {
            "run_id": run.info.run_id,
            "model_name": params.model.model_name,
            "train_rows": len(train_df), "test_rows": len(test_df),
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "train_indices_sha256": hashlib.sha256(
                np.asarray(train_df.index, dtype=np.int64).tobytes()).hexdigest(),
            "test_indices_sha256": hashlib.sha256(
                np.asarray(test_df.index, dtype=np.int64).tobytes()).hexdigest(),
            "tracking_uri": mlflow.get_tracking_uri(),
        }
        _save_artifacts(params, model, transformer, metrics, manifest)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: TrainingPipelineConfig) -> None:
    """Validate structured configuration and execute the training pipeline."""
    train_pipeline(OmegaConf.to_object(cfg))


if __name__ == "__main__":
    main()
