from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    init_from: str = "scratch"
    n_layer: int = 16
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 1024
    bias: bool = False
    dropout: float = 0.0
    learning_rate: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 1e-1
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 3e-5
    meta_path: str = "data/lichess/meta.pkl"
    vocab_size: int = 0


@dataclass
class DataConfig:
    dataset_path: str = "data/lichess"
    file_path: str = "lichess_6gb_blocks.zip"
    batch_size: int = 128
    num_workers: int = 8


@dataclass
class TrainConfig:
    max_epochs: int = 5
    val_check_interval: float = 0.2
    log_every_n_steps: int = 1
    overfit_batches: int = 0
    checkpoint_path: str = "checkpoints/"
    checkpoint_interval: int = 10000
    wandb_project: str = "chessgpt"
    wandb_tags: List[str] = field(default_factory=list)
    gradient_accumulation_steps: int = 10
