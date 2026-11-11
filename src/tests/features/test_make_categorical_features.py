
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from satisfaction.features.build_features import TargetEncoder, build_categorical_pipeline


def test_target_encoder_smoothing():
    x = pd.DataFrame({"category": ["a", "a", "b"]}, index=[7, 8, 9])
    y = pd.Series([1, 3, 5], index=[90, 80, 70])
    encoder = TargetEncoder(smoothing=2).fit(x, y)
    assert encoder.global_mean_ == 3
    assert encoder.mappings_["category"]["a"] == 2.5
    assert encoder.mappings_["category"]["b"] == pytest.approx(11 / 3)
    values = encoder.transform(pd.DataFrame({"category": ["a", "b", "new", np.nan]}))
    np.testing.assert_allclose(values.to_numpy().ravel(), [2.5, 11 / 3, 3, 3])


@pytest.mark.parametrize("kind", ["onehot", "target"])
def test_categorical_pipeline(kind):
    data = pd.DataFrame({"a": ["red", "blue", np.nan, "red"]})
    pipeline = build_categorical_pipeline(kind)
    train = pipeline.fit_transform(data, [1, 2, 4, 5])
    test = pipeline.transform(pd.DataFrame({"a": ["unknown", np.nan]}))
    assert train.shape[1] == test.shape[1]
    values = test.toarray() if hasattr(test, "toarray") else np.asarray(test)
    assert np.isfinite(values).all()


def test_encoder_validation():
    with pytest.raises(ValueError, match="Unknown categorical encoder"):
        build_categorical_pipeline("bad")
    with pytest.raises(NotFittedError):
        TargetEncoder().transform([["a"]])
    with pytest.raises(ValueError, match="non-negative"):
        TargetEncoder(-1).fit([["a"]], [1])
    with pytest.raises(ValueError, match="one finite target"):
        TargetEncoder().fit([["a"]], [1, 2])


def test_target_encoder_zero_smoothing():
    encoded = TargetEncoder(0).fit_transform([["a"], ["a"], ["b"]], [1, 3, 5])
    np.testing.assert_allclose(np.asarray(encoded).ravel(), [2, 2, 5])
