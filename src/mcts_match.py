"""Play a match between two MCTS engines (white vs black)."""
import argparse
import chess
import chess.pgn as pgn

from src.mcts import ChessState, MCTSEngine
from src.train import set_seed


def run_match(
    player1_config: str,
    player2_config: str,
    player1_checkpoint: str | None,
    player2_checkpoint: str | None,
    player1_exploration_weight: float,
    player2_exploration_weight: float,
    num_games: int,
    time_limit: float,
    node_batch_size: int,
    seed: int,
    verbose: bool,
) -> dict:
    """Play num_games games, alternating colors between player1 and player2."""
    set_seed(seed)
    player1_engine = MCTSEngine(
        config_path=player1_config,
        checkpoint_path=player1_checkpoint,
        exploration_weight=player1_exploration_weight,
    )
    player2_engine = MCTSEngine(
        config_path=player2_config,
        checkpoint_path=player2_checkpoint,
        exploration_weight=player2_exploration_weight,
    )

    stats = {
        "player1_wins": 0,
        "player2_wins": 0,
        "draws": 0,
        "games": 0,
    }

    for game_idx in range(num_games):
        player1_is_white = (game_idx % 2 == 0)
        white_engine = player1_engine if player1_is_white else player2_engine
        black_engine = player2_engine if player1_is_white else player1_engine

        board = chess.Board()
        state = ChessState(board)
        plies = 0
        if verbose:
            print(f"Game {game_idx + 1}/{num_games} | player1 is white: {player1_is_white}")
            print(board)

        while not state.is_terminal():
            plies += 1
            engine = white_engine if state.board.turn == chess.WHITE else black_engine
            move, eval_score = engine.mcts_move(
                state,
                node_batch_size=node_batch_size,
                time_limit=time_limit,
                verbose=verbose,
            )
            if verbose:
                print('Making move:', move, 'eval:', round(eval_score, 3), 'ply:', plies)
            state = state.make_move(move)
            if verbose:
                print(state.board)

        board = state.board
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            stats["draws"] += 1
        elif outcome.winner == chess.WHITE:
            if player1_is_white:
                stats["player1_wins"] += 1
            else:
                stats["player2_wins"] += 1
        else:
            if player1_is_white:
                stats["player2_wins"] += 1
            else:
                stats["player1_wins"] += 1
        stats["games"] += 1

        if verbose:
            print('Outcome:', outcome)
            game = pgn.Game().from_board(board)
            pgn_string = game.accept(pgn.StringExporter(headers=True, variations=True, comments=True))
            print(pgn_string)

    if verbose:
        print('Calls (player1):', player1_engine.calls_to_eval, 'Calls (player2):', player2_engine.calls_to_eval)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a head-to-head MCTS match.")
    parser.add_argument("--player1-config", default="configs/test_config.py")
    parser.add_argument("--player2-config", default="configs/test_config.py")
    parser.add_argument("--player1-checkpoint", default=None)
    parser.add_argument("--player2-checkpoint", default=None)
    parser.add_argument("--player1-exploration-weight", type=float, default=0.1)
    parser.add_argument("--player2-exploration-weight", type=float, default=0.1)
    parser.add_argument("--num-games", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=2.0)
    parser.add_argument("--node-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    stats = run_match(
        player1_config=args.player1_config,
        player2_config=args.player2_config,
        player1_checkpoint=args.player1_checkpoint,
        player2_checkpoint=args.player2_checkpoint,
        player1_exploration_weight=args.player1_exploration_weight,
        player2_exploration_weight=args.player2_exploration_weight,
        num_games=args.num_games,
        time_limit=args.time_limit,
        node_batch_size=args.node_batch_size,
        seed=args.seed,
        verbose=not args.quiet,
    )
    print("Stats:", stats)


if __name__ == "__main__":
    main()
