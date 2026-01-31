import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import chess
import pandas as pd
import torch

from bin_predictor import BinPredictor
from src.mcts import ChessState, MCTSEngine
from src.models.model import ChessTransformer


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


def best_move_from_fen(model: ChessTransformer, board: chess.Board, device: str) -> chess.Move:
    legal_moves = list(board.legal_moves)
    fen_list = []
    three_fold_indices = []
    for i, move in enumerate(legal_moves):
        board.push(move)
        fen_list.append(board.fen())
        if board.can_claim_draw() or board.is_stalemate():
            three_fold_indices.append(i)
        board.pop()

    with torch.no_grad():
        evals = model.compute_avg_bin_index_from_fens(fen_list, device=device)
        mid = (model.bin_predictor.total_num_bins + 1) / 2
        for i in three_fold_indices:
            if board.turn:
                evals[i] = min(mid, evals[i])
            else:
                evals[i] = max(mid, evals[i])

    best_idx = evals.argmax().item() if board.turn else evals.argmin().item()
    return legal_moves[best_idx]


def pick_move(
    board: chess.Board,
    *,
    model: Optional[ChessTransformer],
    device: str,
    use_mcts: bool,
    mcts_engine: Optional[MCTSEngine],
    mcts_time_limit: float,
) -> chess.Move:
    if use_mcts:
        if mcts_engine is None:
            raise RuntimeError("MCTS engine is required when use_mcts is True.")
        state = ChessState(board.copy())
        move, _ = mcts_engine.mcts_move(
            state,
            node_batch_size=1,
            time_limit=mcts_time_limit,
            verbose=False,
        )
        return move
    if model is None:
        raise RuntimeError("Model is required when use_mcts is False.")
    return best_move_from_fen(model, board, device)


def iter_puzzles(csv_path: str, limit: Optional[int]) -> Iterable[Tuple[str, str, int]]:
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)
    for _, row in df.iterrows():
        fen = str(row["FEN"])
        moves = str(row["Moves"])
        rating = int(row["Rating"])
        yield fen, moves, rating


def evaluate_puzzle(
    model: Optional[ChessTransformer],
    fen: str,
    moves_str: str,
    device: str,
    *,
    use_mcts: bool,
    mcts_engine: Optional[MCTSEngine],
    mcts_time_limit: float,
) -> bool:
    board = chess.Board(fen)
    moves = moves_str.split()

    if not moves:
        return False

    # Apply the first move (puzzle setup move)
    first_move = chess.Move.from_uci(moves[0])
    if first_move not in board.legal_moves:
        return False
    board.push(first_move)

    # Engine must match moves at indices 1,3,5,... (i.e., every other move after the first)
    for idx in range(1, len(moves), 2):
        expected = chess.Move.from_uci(moves[idx])
        if expected not in board.legal_moves:
            return False

        predicted = pick_move(
            board,
            model=model,
            device=device,
            use_mcts=use_mcts,
            mcts_engine=mcts_engine,
            mcts_time_limit=mcts_time_limit,
        )
        if predicted != expected:
            return False

        board.push(expected)

        # Apply the opponent reply if present (indices 2,4,6,...)
        if idx + 1 < len(moves):
            reply = chess.Move.from_uci(moves[idx + 1])
            if reply not in board.legal_moves:
                return False
            board.push(reply)

    return True

@dataclass
class PuzzleEvalStats:
    bucket_stats: Dict[str, Dict[str, int]]
    skipped: int


def evaluate_puzzles(
    model: Optional[ChessTransformer],
    csv_path: str,
    limit: Optional[int],
    device: str,
    *,
    use_mcts: bool,
    mcts_engine: Optional[MCTSEngine],
    mcts_time_limit: float,
) -> PuzzleEvalStats:
    bucket_edges = list(range(1000, 3001, 200))  # 1000..3000
    bucket_stats = {f"{bucket_edges[i]}-{bucket_edges[i+1]}": {"total": 0, "solved": 0}
                    for i in range(len(bucket_edges) - 1)}
    skipped = 0
    evaluated = 0

    def bucket_label(rating: int) -> Optional[str]:
        if rating < 1000 or rating > 3000:
            return None
        if rating == 3000:
            return "2800-3000"
        idx = (rating - 1000) // 200
        low = 1000 + 200 * idx
        high = low + 200
        return f"{low}-{high}"

    def print_interim_stats() -> None:
        print("Interim accuracy by rating bucket:")
        for label in bucket_stats:
            total = bucket_stats[label]["total"]
            solved = bucket_stats[label]["solved"]
            acc = solved / max(1, total)
            print(f"  {label}: {solved}/{total} = {acc:.4f}")

    for fen, moves, rating in iter_puzzles(csv_path, limit):
        label = bucket_label(rating)
        if label is None:
            skipped += 1
            continue

        bucket_stats[label]["total"] += 1
        if evaluate_puzzle(
            model,
            fen,
            moves,
            device,
            use_mcts=use_mcts,
            mcts_engine=mcts_engine,
            mcts_time_limit=mcts_time_limit,
        ):
            bucket_stats[label]["solved"] += 1

        evaluated += 1
        if evaluated % 100 == 0:
            print(f"Evaluated {evaluated} puzzles...")
        if evaluated % 100 == 0:
            print_interim_stats()

    return PuzzleEvalStats(bucket_stats=bucket_stats, skipped=skipped)


def run(
    config_path: str,
    csv_path: str,
    limit: Optional[int],
    *,
    use_mcts: bool,
    mcts_time_limit: float,
) -> None:
    config = load_config_from_path(config_path)
    trainer = config.trainer
    model_cfg = config.model

    experiment_root = Path(trainer.experiment_dir) / config.exp_name
    checkpoint_dir = experiment_root / "checkpoints"
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    mcts_engine = None
    model = None
    if use_mcts:
        mcts_engine = MCTSEngine(
            config_path=config_path,
            device=trainer.device,
            checkpoint_path=checkpoint_path,
        )
    else:
        bin_predictor = BinPredictor()
        model = model_cfg.create_model(bin_predictor=bin_predictor, device=trainer.device)
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        model.load_state_dict(checkpoint["transformer_state_dict"])
        model.eval()

    stats = evaluate_puzzles(
        model,
        csv_path,
        limit,
        trainer.device,
        use_mcts=use_mcts,
        mcts_engine=mcts_engine,
        mcts_time_limit=mcts_time_limit,
    )

    print("Accuracy by rating bucket:")
    for label in stats.bucket_stats:
        total = stats.bucket_stats[label]["total"]
        solved = stats.bucket_stats[label]["solved"]
        acc = solved / max(1, total)
        print(f"  {label}: {solved}/{total} = {acc:.4f}")

    if stats.skipped > 0:
        print(f"Skipped (out of 1000-3000): {stats.skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, help="Path to a python config file with get_config()")
    parser.add_argument("--csv_path", default=r"C:\Users\Ahmad-personal\PycharmProjects\chess_stockfish_evals_v2\data\lichess_db_puzzle.csv", help="Path to puzzle CSV")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of puzzles")
    parser.add_argument("--use-mcts", action="store_true", help="Use MCTS to select moves")
    parser.add_argument("--mcts-time-limit", type=float, default=0.5, help="Time limit per MCTS move (seconds)")
    args = parser.parse_args()

    run(
        args.config_path,
        args.csv_path,
        args.limit,
        use_mcts=args.use_mcts,
        mcts_time_limit=args.mcts_time_limit,
    )
