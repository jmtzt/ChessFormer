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
