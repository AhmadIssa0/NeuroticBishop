"""Fix checkpoints trained with a different BinPredictor bin count."""
import argparse
import importlib.util
from pathlib import Path
from typing import Optional

import torch

from src.bin_predictor import BinPredictor


def load_config_from_path(config_path: str):
    config_file = Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location("config_module", config_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load config from: {config_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_config"):
        raise AttributeError(f"{config_file} must define get_config()")
    return module.get_config()


def find_latest_checkpoint(search_dir: Path) -> Optional[str]:
    candidates = list(search_dir.glob("checkpoint_*.pth"))
    if not candidates:
        return None

    def extract_step(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except ValueError:
            return -1

    best = max(candidates, key=extract_step)
    return str(best)


def fix_checkpoint(
    config_path: str,
    checkpoint_path: Optional[str],
    output_path: Optional[str],
) -> str:
    """Load a checkpoint and skip the bin projection layer, then save it."""
    config = load_config_from_path(config_path)
    trainer = config.trainer
    model_cfg = config.model

    if checkpoint_path is None:
        experiment_root = Path(trainer.experiment_dir) / config.exp_name
        checkpoint_dir = experiment_root / "checkpoints"
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    if output_path is None:
        in_path = Path(checkpoint_path)
        output_path = str(in_path.with_name(f"{in_path.stem}_fixed{in_path.suffix}"))

    model = model_cfg.create_model(bin_predictor=BinPredictor(), device=trainer.device)
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
    state_dict = checkpoint["transformer_state_dict"]
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("proj_to_bin_predictor_logits.")
    }
    model.load_state_dict(filtered_state, strict=False)

    updated_checkpoint = dict(checkpoint)
    updated_checkpoint["transformer_state_dict"] = model.state_dict()
    torch.save(updated_checkpoint, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix checkpoint bin projection layer.")
    parser.add_argument("--config_path", required=True, help="Path to a python config file with get_config()")
    parser.add_argument("--checkpoint_path", default=None, help="Path to a checkpoint file to fix")
    parser.add_argument("--output_path", default=None, help="Where to save the fixed checkpoint")
    args = parser.parse_args()

    output_path = fix_checkpoint(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
    )
    print("Saved fixed checkpoint to:", output_path)


if __name__ == "__main__":
    main()
