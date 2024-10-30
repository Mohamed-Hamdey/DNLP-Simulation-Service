from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    max_length: int = 128
    embedding_dim: int = 768
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.3
    num_heads: int = 8
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    model_path: str = "best_model.pt"
    device: Optional[str] = None
    output_dim = None