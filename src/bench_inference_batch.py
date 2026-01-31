"""Benchmark model inference time vs batch size."""
import argparse
import importlib.util
import time
from pathlib import Path
from typing import Iterable, Optional

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


def parse_batch_sizes(value: str) -> list[int]:
    sizes = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        sizes.append(int(part))
    if not sizes:
        raise ValueError("No batch sizes provided.")
    return sizes


def _maybe_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def bench_batch_sizes(
    model,
    device: str,
    batch_sizes: Iterable[int],
    repeats: int,
    warmup: int,
    use_autocast: bool,
) -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    for batch_size in batch_sizes:
        fens = [fen] * batch_size

        for _ in range(warmup):
            with torch.no_grad():
                if use_autocast and device.startswith("cuda"):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _ = model.compute_white_win_prob_from_fen(fens, device=device)
                else:
                    _ = model.compute_white_win_prob_from_fen(fens, device=device)
        _maybe_sync(device)

        total = 0.0
        for _ in range(repeats):
            start = time.perf_counter()
            with torch.no_grad():
                if use_autocast and device.startswith("cuda"):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _ = model.compute_white_win_prob_from_fen(fens, device=device)
                else:
                    _ = model.compute_white_win_prob_from_fen(fens, device=device)
            _maybe_sync(device)
            total += (time.perf_counter() - start)

        avg_ms = (total / repeats) * 1000.0
        print(f"batch_size={batch_size} avg_ms={avg_ms:.3f}")


def run(
    config_path: str,
    batch_sizes: Iterable[int],
    repeats: int,
    warmup: int,
    use_autocast: bool,
) -> None:
    config = load_config_from_path(config_path)
    trainer = config.trainer
    model_cfg = config.model

    experiment_root = Path(trainer.experiment_dir) / config.exp_name
    checkpoint_dir = experiment_root / "checkpoints"
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    model = model_cfg.create_model(bin_predictor=BinPredictor(), device=trainer.device)
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
    model.load_state_dict(checkpoint["transformer_state_dict"])
    model.eval()

    bench_batch_sizes(
        model=model,
        device=trainer.device,
        batch_sizes=batch_sizes,
        repeats=repeats,
        warmup=warmup,
        use_autocast=use_autocast,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference time vs batch size.")
    parser.add_argument("--config_path", required=True, help="Path to a python config file with get_config()")
    parser.add_argument("--batch-sizes", default="1,4,16,64,256,1024", help="Comma-separated batch sizes")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-autocast", action="store_true", help="Disable autocast")
    args = parser.parse_args()

    run(
        config_path=args.config_path,
        batch_sizes=parse_batch_sizes(args.batch_sizes),
        repeats=args.repeats,
        warmup=args.warmup,
        use_autocast=not args.no_autocast,
    )


if __name__ == "__main__":
    main()
