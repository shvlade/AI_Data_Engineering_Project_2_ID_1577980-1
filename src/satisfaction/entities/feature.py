
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FeatureConfig:
    
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: str
    categorical_encoder: str = field(default="onehot")
    features_to_drop: Optional[List[str]] = None
