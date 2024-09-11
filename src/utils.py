import math
import pickle
from dataclasses import fields
from pathlib import Path
from typing import Any


def filter_config(config_dict: dict, config_class: Any) -> dict:
    valid_keys = {field.name for field in fields(config_class)}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    return filtered_config


def get_metadata(meta_path: str) -> dict:
    meta = None
    if Path(meta_path).exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

    if not meta:
        raise FileNotFoundError(f"Meta file not found at {meta_path}")

    return meta


def get_learning_rate(learning_rate, warmup_iters, lr_decay_iters, min_lr, it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    lr = min_lr + coeff * (learning_rate - min_lr)
    return lr
