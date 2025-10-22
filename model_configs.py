"""
Model configuration management for GNN training.
Stores and manages model configurations for different datasets.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for a GNN model."""
    hidden_channels: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


class ConfigManager:
    """Manages model configurations for different datasets."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def get_default_configs(self) -> Dict[str, ModelConfig]:
        """Get default configurations for each dataset."""
        return {
            "BBBP": ModelConfig(
                hidden_channels=128,
                num_layers=3,
                dropout=0.1,
                learning_rate=0.001,
                weight_decay=1e-5,
                batch_size=256,
                epochs=100
            ),
            "BACE": ModelConfig(
                hidden_channels=64,
                num_layers=2,
                dropout=0.2,
                learning_rate=0.01,
                weight_decay=1e-6,
                batch_size=128,
                epochs=80
            ),
            "HIV": ModelConfig(
                hidden_channels=256,
                num_layers=4,
                dropout=0.15,
                learning_rate=0.0005,
                weight_decay=1e-7,
                batch_size=512,
                epochs=120
            )
        }
    
    def save_config(self, dataset_name: str, config: ModelConfig) -> None:
        """Save configuration for a dataset."""
        config_path = os.path.join(self.config_dir, f"{dataset_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    def load_config(self, dataset_name: str) -> Optional[ModelConfig]:
        """Load configuration for a dataset."""
        config_path = os.path.join(self.config_dir, f"{dataset_name}_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return ModelConfig.from_dict(config_dict)
        return None
    
    def get_config(self, dataset_name: str) -> ModelConfig:
        """Get configuration for a dataset, creating default if not exists."""
        config = self.load_config(dataset_name)
        if config is None:
            default_configs = self.get_default_configs()
            config = default_configs.get(dataset_name, ModelConfig())
            self.save_config(dataset_name, config)
        return config
    
    def update_config(self, dataset_name: str, **kwargs) -> ModelConfig:
        """Update configuration for a dataset with new parameters."""
        config = self.get_config(dataset_name)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        self.save_config(dataset_name, config)
        return config
    
    def list_configs(self) -> Dict[str, ModelConfig]:
        """List all available configurations."""
        configs = {}
        for filename in os.listdir(self.config_dir):
            if filename.endswith('_config.json'):
                dataset_name = filename.replace('_config.json', '')
                config = self.load_config(dataset_name)
                if config:
                    configs[dataset_name] = config
        return configs
    
    def initialize_default_configs(self) -> None:
        """Initialize default configurations for all datasets."""
        default_configs = self.get_default_configs()
        for dataset_name, config in default_configs.items():
            if not os.path.exists(os.path.join(self.config_dir, f"{dataset_name}_config.json")):
                self.save_config(dataset_name, config)
