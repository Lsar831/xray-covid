"""Load and access project configuration from a single YAML file."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative_path: str) -> Path:
    """Resolve a config-relative path to an absolute path under the project root."""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


@dataclass
class Paths:
    raw_metadata: Path
    raw_images_dir: Path
    processed_dir: Path
    splits_dir: Path

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Paths":
        d = cfg["data"]
        return cls(
            raw_metadata=resolve_path(d["raw_metadata"]),
            raw_images_dir=resolve_path(d["raw_images_dir"]),
            processed_dir=resolve_path(d["processed_dir"]),
            splits_dir=resolve_path(d["splits_dir"]),
        )
