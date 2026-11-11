
from typing import Any
from dataclasses import dataclass, field


@dataclass
class GBRConfig:
    
    _target_: str = "sklearn.ensemble.GradientBoostingRegressor"
    n_estimators: int = 150
    max_depth: int = 3
    learning_rate: float = 0.05
    random_state: int = 42


@dataclass
class LinRegConfig:
    
    _target_: str = "sklearn.linear_model.LinearRegression"
    fit_intercept: bool = True


@dataclass
class ModelConfig:
    
    model_name: str
    model_params: Any


@dataclass
class LinearModelConfig(ModelConfig):
    
    model_name: str = "lin-reg"
    model_params: LinRegConfig = field(default_factory=LinRegConfig)


@dataclass
class GradientBoostingModelConfig(ModelConfig):
   
    model_name: str = "gbr"
    model_params: GBRConfig = field(default_factory=GBRConfig)
