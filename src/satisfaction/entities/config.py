
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from .models import ModelConfig, LinearModelConfig, GradientBoostingModelConfig
from .feature import FeatureConfig


@dataclass
class DatasetConfig:
    
    input_data_path: str


@dataclass
class SplittingConfig:
    
    test_size: float = field(default=0.25)
    random_state: int = field(default=42)


@dataclass
class TrainingPipelineConfig:
   
    model: ModelConfig
    dataset: DatasetConfig
    feature: FeatureConfig
    split: SplittingConfig


cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainingPipelineConfig)
cs.store(group="model", name="lr_schema", node=LinearModelConfig)
cs.store(group="model", name="gbr_schema", node=GradientBoostingModelConfig)
