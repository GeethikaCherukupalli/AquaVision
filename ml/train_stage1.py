"""Train the AquaVision Stage 1 models from the repository configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .resnet.train import train_resnet
from .unet.train import train_unet


def load_config(path: str | None, section: str) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    values = document.get(section, {})
    if not isinstance(values, dict):
        raise ValueError(f"Configuration section '{section}' must be a mapping")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an AquaVision Stage 1 model")
    parser.add_argument("--model", choices=("resnet", "unet"), required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", help="Path to stage1 YAML configuration")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    config = load_config(args.config, args.model)
    if args.model == "resnet":
        result = train_resnet(str(dataset_root), config)
    else:
        result = train_unet(str(dataset_root), config)

    print(f"Training complete: {result['artifact_dir']}")
    print(f"Metadata: {result['metadata_path']}")


if __name__ == "__main__":
    main()
