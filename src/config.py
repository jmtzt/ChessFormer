from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    init_from: str
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    bias: bool
    dropout: float
    learning_rate: float
    beta1: float
    beta2: float
    weight_decay: float
    warmup_iters: int
    lr_decay_iters: int
    min_lr: float
    meta_path: str


@dataclass
class DataConfig:
    dataset_path: str
    file_path: str
    batch_size: int
    num_workers: int


@dataclass
class TrainConfig:
    max_epochs: int
    val_check_interval: float
    log_every_n_steps: int
    overfit_batches: int
    checkpoint_path: str
    checkpoint_interval: int
    wandb_project: str
    wandb_tags: List[str]
    gradient_accumulation_steps: int


ChessGPTConfig = ModelConfig(
    init_from="scratch",
    n_layer=16,
    n_head=8,
    n_embd=512,
    block_size=1024,
    bias=False,
    dropout=0.0,
    learning_rate=6e-4,
    beta1=0.9,
    beta2=0.95,
    weight_decay=1e-1,
    warmup_iters=2000,
    lr_decay_iters=600000,
    min_lr=3e-5,
    meta_path="data/lichess/meta.pkl",
)

ChessGPTDataConfig = DataConfig(
    dataset_path="data/lichess",
    file_path="lichess_6gb_blocks.zip",
    batch_size=128,
    num_workers=8,
)

ChessGPTTrainConfig = TrainConfig(
    max_epochs=5,
    val_check_interval=0.2,
    log_every_n_steps=1,
    overfit_batches=0,
    checkpoint_path="checkpoints/",
    checkpoint_interval=10000,
    wandb_project="chessgpt",
    wandb_tags=["initial_run"],
    gradient_accumulation_steps=10,
)
