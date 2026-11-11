
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from satisfaction.data.make_dataset import read_data
from satisfaction.features.build_features import engineer_features, make_features
from satisfaction.models.predict import predict_model


def run_inference(
    model_path: str, transformer_path: str, input_path: str, output_path: str,
) -> pd.DataFrame:
    """Load trusted pickles and score raw orders without requiring review_score."""
    with open(model_path, "rb") as stream:
        model = pickle.load(stream)
    with open(transformer_path, "rb") as stream:
        transformer = pickle.load(stream)
    data = read_data(input_path)
    if "order_id" not in data:
        raise ValueError("Input CSV must contain order_id")
    features = make_features(transformer, engineer_features(data))
    result = pd.DataFrame({
        "order_id": data["order_id"].to_numpy(),
        "predicted_review_score": np.clip(predict_model(model, features), 1, 5),
    })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main() -> None:
    """Parse the four required paths and produce the predictions CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "transformer-path", "input-path", "output-path"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    result = run_inference(
        args.model_path, args.transformer_path, args.input_path, args.output_path
    )
    print(f"Saved {len(result)} predictions to {args.output_path}")


if __name__ == "__main__":
    main()
