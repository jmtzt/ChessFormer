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


def get_learning_rate(warmup_iters, warmdown_iters, lr_decay_iters, it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    # 2) constant lr for a while
    elif it < lr_decay_iters - warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
