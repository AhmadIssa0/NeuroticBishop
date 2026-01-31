
import chess
import torch
from src.models.model import ChessTransformer
import chess.pgn as pgn
import random
from bin_predictor import BinPredictor
from torch.cuda.amp import autocast
import matplotlib
import argparse
import importlib.util
from pathlib import Path
from typing import Optional

matplotlib.use('agg')

device = 'cuda'

bin_pred = BinPredictor()


def best_move_from_fen(model, board):
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
        evals = model.compute_white_win_prob_from_fen(fen_list, device=device)
        # evals = model.compute_avg_bin_index_from_fens(fen_list, device=device)

        for i in three_fold_indices:
            if board.turn:
                evals[i] = min(0.5, evals[i])
                # evals[i] = min((bin_pred.total_num_bins + 1) / 2, evals[i])
            else:
                evals[i] = max(0.5, evals[i])
                # evals[i] = max((bin_pred.total_num_bins + 1) / 2, evals[i])
    if board.turn:
        best_move_idx = evals.argmax().item()
    else:
        best_move_idx = evals.argmin().item()
    best_move = legal_moves[best_move_idx]
    print('All evals:', list(zip(evals, legal_moves)))
    return best_move, evals[best_move_idx].item()


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


def run(config_path: str):
    config = load_config_from_path(config_path)
    trainer = config.trainer
    model_cfg = config.model

    experiment_root = Path(trainer.experiment_dir) / config.exp_name
    checkpoint_dir = experiment_root / "checkpoints"

    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Initialize a board from a FEN string
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Starting position
    # fen = "6k1/1pp2R2/2n3P1/p3p1qP/1P5b/P1P5/2K1R3/8 b - - 6 56"
    board = chess.Board(fen)

    chess_network = ChessTransformer(
        bin_predictor=BinPredictor(),
        d_model=model_cfg.d_model,
        num_layers=model_cfg.num_layers,
        nhead=model_cfg.nhead,
        dim_feedforward=model_cfg.resolved_dim_feedforward(),
        norm_first=model_cfg.norm_first,
        use_gelu=model_cfg.use_gelu,
    ).to(device=trainer.device)

    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
    print("Loaded checkpoint:", checkpoint_path)
    chess_network.load_state_dict(checkpoint['transformer_state_dict'])
    chess_network.eval()

    evals = [0.5, 0.5]
    stds = [0.0, 0.0]
    board.push(random.choice(list(board.legal_moves)))
    board.push(random.choice(list(board.legal_moves)))
    ply = 2

    while not board.is_game_over():
        # with torch.no_grad():
        #     probs = transformer.compute_bin_probabilities_from_fens([board.fen()], device)[0].tolist()
        #     indices = range(len(probs))
        #     plt.bar(indices, probs)
        #     plt.savefig(f'plots/probabilities_{ply}.png')
        #     plt.close()
        #     ply += 1
        best_move, eval = best_move_from_fen(chess_network, board)
        print('Making move:', best_move, 'eval after making move:', eval)
        print(board)
        board.push(best_move)
        evals.append(eval)
        # with torch.no_grad():
        #     index_means, index_stds = chess_network.compute_bin_index_means_and_stds_from_fens([board.fen()], device)
            # stds.append(index_stds[0].item() / (bin_pred.total_num_bins - 1))

    print('Outcome:', board.outcome())
    # print(pgn.Game.from_board(board))

    game = pgn.Game().from_board(board)
    node = game
    for i, eval in enumerate(evals):
        node = node.next()
        node.comment = f"[%eval {eval:.4f}]"

    # Export to PGN
    pgn_string = game.accept(pgn.StringExporter(headers=True, variations=True, comments=True))

    print(pgn_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, help="Path to a python config file with get_config()")
    args = parser.parse_args()

    with autocast():
        run(args.config_path)

