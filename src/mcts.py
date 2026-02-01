"""Monte Carlo Tree Search engine with a neural evaluator for chess."""
import random
import math
import chess
import torch
import chess.pgn as pgn
import time
import threading
from typing import Optional, List, Dict, Tuple, Iterable
from src.bin_predictor import BinPredictor
from torch.cuda.amp import autocast
from itertools import accumulate
from src.train import set_seed
import importlib.util
from pathlib import Path


class ChessState:
    """Immutable wrapper around a chess.Board representing a game state."""

    def __init__(self, board: Optional[chess.Board] = None) -> None:
        self.board = chess.Board() if board is None else board

    def get_moves(self) -> List[chess.Move]:
        return list(self.board.legal_moves)

    def make_move(self, move: chess.Move) -> "ChessState":
        new_board = self.board.copy()
        new_board.push(move)
        return ChessState(new_board)

    def is_terminal(self) -> bool:
        return self.board.is_game_over(claim_draw=False) or self.board.is_repetition() or self.board.is_fifty_moves()

    def is_win(self, player_color: str) -> bool:
        """Return True if the given player_color has a win in the current terminal state."""
        if self.board.is_checkmate():
            return True if (player_color == 'WHITE' and self.board.turn == chess.WHITE) or \
                           (player_color == 'BLACK' and self.board.turn == chess.BLACK) else False
        return False

    @property
    def player(self) -> str:
        return 'WHITE' if self.board.turn else 'BLACK'


class UCTNode:
    """Node in the UCT search tree."""

    def __init__(
        self,
        device: str,
        exploration_weight: float,
        move: Optional[chess.Move] = None,
        move_idx: Optional[int] = None,
        parent: Optional["UCTNode"] = None,
        state: Optional[ChessState] = None,
    ) -> None:
        self.move: chess.Move = move
        self.move_idx = move_idx  # index of the move in parent's legal_moves list
        self.parent: Optional[UCTNode] = parent
        self.state: ChessState = state
        self.legal_moves: List[chess.Move] = state.get_moves()
        self.is_expanded = False  # a node is expanded if the initial evals of its children have been set
        self.applied_virtual_loss_count = 0
        self.device = device
        self.exploration_weight = exploration_weight
        if parent is None:
            self.plies_from_root = 0
        else:
            self.plies_from_root = parent.plies_from_root + 1

        self.children: Dict[int, "UCTNode"] = {}  # Dict[move_idx, UCTNode]
        # For the values below, we use absolute values in [0, 1], where '1' means white wins
        self.child_total_value = torch.zeros([len(self.legal_moves)], dtype=torch.float32, device=device)
        self.child_number_visits = torch.zeros([len(self.legal_moves)], dtype=torch.float32, device=device)
        self.child_priors = torch.full(
            [len(self.legal_moves)],
            1.0 / max(1, len(self.legal_moves)),
            dtype=torch.float32,
            device=device,
        )

    @property
    def total_value(self) -> float:
        """Value of current node from white's perspective."""
        if self.is_root():
            best_move_idx = self.best_move_idx()
            return self.child_total_value[best_move_idx].item()
        return self.parent.child_total_value[self.move_idx].item()

    @total_value.setter
    def total_value(self, value: float) -> None:
        self.parent.child_total_value[self.move_idx] = value

    @property
    def visits(self) -> float:
        if self.is_root():
            return self.child_number_visits.sum().item()
        return self.parent.child_number_visits[self.move_idx].item()

    @visits.setter
    def visits(self, value: float) -> None:
        self.parent.child_number_visits[self.move_idx] = value

    def is_root(self) -> bool:
        return self.parent is None

    def add_child(self, child_node: "UCTNode") -> None:
        self.children[child_node.move_idx] = child_node

    def best_move_idx(self) -> int:
        """Return index of the best move according to visits and value."""
        max_visits = torch.max(self.child_number_visits).item()
        child_q = self.child_total_value / (1.0 + self.child_number_visits)
        if self.state.board.turn == chess.BLACK:
            child_q = 1.0 - child_q
        child_q = child_q * (self.child_number_visits == max_visits).float()
        return torch.argmax(child_q).item()

    def update(self, result: float) -> None:
        self.visits += 1
        self.total_value += result

    def uct_select_child(self) -> "UCTNode":
        assert self.is_expanded

        child_q = self.child_total_value / (1.0 + self.child_number_visits)
        if self.state.board.turn == chess.BLACK:
            child_q = 1.0 - child_q

        parent_visits = self.visits
        # print(f"weight: {self.exploration_weight}, priors: {self.child_priors}, visits: {parent_visits}, q: {child_q},")
        child_u = self.exploration_weight * self.child_priors * torch.sqrt(
            torch.tensor(parent_visits, device=self.device) + 1.0
        ) / (1.0 + self.child_number_visits)

        best_move_idx = torch.argmax(child_q + child_u).item()
        return self.maybe_add_child(best_move_idx)

    def eval_player_perspective(self) -> float:
        white_eval = self.total_value / (1 + self.visits)
        if self.state.board.turn == chess.BLACK:
            return 1.0 - white_eval
        return white_eval

    def eval_based_on_children(self) -> float:
        child_q = self.child_total_value / (1.0 + self.child_number_visits)
        if self.state.board.turn == chess.WHITE:
            return child_q.max().item()
        else:
            return child_q.min().item()

    def add_virtual_loss(self) -> None:
        current = self
        sgn = 1.0 if self.state.board.turn == chess.WHITE else -1.0
        while current is not None and current.parent is not None:  # parent is None for the root!
            if not current.is_root():
                current.total_value += (sgn + 1.0) / 2.0 * 0.3
                current.visits += 1
            current.applied_virtual_loss_count += 1
            current = current.parent
            sgn *= -1

    def revert_virtual_loss(self) -> None:
        current = self
        sgn = 1.0 if self.state.board.turn == chess.WHITE else -1.0
        while current is not None and current.parent is not None:
            if not current.is_root():
                current.total_value -= (sgn + 1.0) / 2.0 * 0.3
                current.visits -= 1
            current.applied_virtual_loss_count -= 1
            current = current.parent
            sgn *= -1

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def maybe_add_child(self, move_idx: int) -> "UCTNode":
        if move_idx in self.children:
            return self.children[move_idx]
        else:
            move = self.legal_moves[move_idx]
            next_state = self.state.make_move(move)
            child_node = UCTNode(
                device=self.device,
                exploration_weight=self.exploration_weight,
                move=move,
                parent=self,
                state=next_state,
                move_idx=move_idx,
            )
            self.add_child(child_node)
            return child_node

    def expand(self, child_evals: torch.Tensor, child_priors: Optional[torch.Tensor] = None) -> None:
        """Expand node by setting initial child evaluations (white perspective)."""
        self.is_expanded = True
        self.child_total_value = child_evals
        if child_priors is not None and child_priors.numel() == self.child_priors.numel():
            self.child_priors = child_priors

    def terminal_state_eval(self) -> float:
        """Evaluate terminal node from white's perspective."""
        board: chess.Board = self.state.board
        assert board.is_game_over(claim_draw=False) or board.is_repetition() or board.is_fifty_moves()
        if board.is_checkmate():  # current player got checkmated
            if board.turn == chess.WHITE:
                return -0.1 + self.plies_from_root * 1e-3
            else:
                return 1.1 - self.plies_from_root * 1e-3
        return 0.5

    def backup(self, result: float) -> None:
        """Back up a value from white's perspective to the root."""
        current = self
        while current is not None and current.parent is not None:
            current.update(result)
            current = current.parent

    def __str__(self, level: int = 0) -> str:
        ret = "\t" * level
        if not self.is_root():
            ret += f"Move: {self.move}, Abs-eval: {self.total_value / (1 + self.visits):.4f}, Visits: {self.visits}, Player after move: {self.state.player}\n"
        else:
            ret += f"Root Node, Abs-eval: {self.eval_based_on_children():.4f}, Visits: {self.visits}\n"

        for child in self.children.values():
            if level <= 0:
                ret += child.__str__(level + 1)
        return ret


def load_config_from_path(config_path: str, **kwargs):
    """Load a config object from a Python file exposing get_config()."""
    config_file = Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location("config_module", config_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load config from: {config_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_config"):
        raise AttributeError(f"{config_file} must define get_config()")
    return module.get_config(**kwargs)


def find_latest_checkpoint(search_dir: Path) -> Optional[str]:
    """Return path to the latest checkpoint in a directory, or None if none exist."""
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


class MCTSEngine:
    """MCTS engine that evaluates positions with a transformer model."""

    def __init__(
        self,
        config_path: str,
        device: str = 'cuda',
        checkpoint_path: Optional[str] = None,
        exploration_weight: float = 1.0,
        reuse_tree: bool = False,
        ponder: bool = False,
        config_kwargs: Optional[dict] = None,
        prior_temperature: float = 0.1,
        root_dirichlet_alpha: float = 0.3,
        root_dirichlet_eps: float = 0.02,
    ) -> None:
        self.device = device
        self.exploration_weight = exploration_weight
        self.reuse_tree = reuse_tree
        self.ponder = ponder
        self._saved_root: Optional[UCTNode] = None
        self._ponder_thread: Optional[threading.Thread] = None
        self._ponder_stop = threading.Event()

        # NEW: PUCT root priors temperature + Dirichlet noise
        self.prior_temperature = float(prior_temperature)
        self.root_dirichlet_alpha = float(root_dirichlet_alpha)
        self.root_dirichlet_eps = float(root_dirichlet_eps)

        config = load_config_from_path(config_path, **(config_kwargs or {}))
        trainer = config.trainer
        model_cfg = config.model

        experiment_root = Path(trainer.experiment_dir) / config.exp_name
        checkpoint_dir = experiment_root / "checkpoints"

        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(checkpoint_dir)
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        transformer = model_cfg.create_model(bin_predictor=BinPredictor(), device=device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        transformer.load_state_dict(checkpoint['transformer_state_dict'])
        transformer.eval()
        self.calls_to_eval = 0
        print(f'Loaded transformer from checkpoint: {checkpoint_path}.')
        transformer = torch.compile(transformer, mode="reduce-overhead")
        # transformer = torch.compile(transformer, mode="max-autotune")
        self.transformer = transformer
        for _ in range(2):
            self.mcts_move(ChessState(chess.Board()), time_limit=2.0, verbose=False)

    def _stop_ponder(self) -> None:
        if self._ponder_thread is not None and self._ponder_thread.is_alive():
            self._ponder_stop.set()
            self._ponder_thread.join()
        self._ponder_stop.clear()

    def _ponder_loop(self, root: UCTNode, node_batch_size: int) -> None:
        while not self._ponder_stop.is_set():
            with torch.no_grad():
                self.mcts(root, node_batch_size=node_batch_size, time_limit=0.25, verbose=False)

    def get_evals_for_all_moves(self, nodes: List[UCTNode]) -> List[torch.Tensor]:
        """Batch-evaluate all child moves for a list of nodes."""
        self.calls_to_eval += 1
        if len(nodes) == 0:
            return []
        node_separations = list(accumulate([len(node.legal_moves) for node in nodes]))[:-1]

        def _get_fens_and_draw_indices(node: UCTNode) -> Tuple[List[str], List[int]]:
            node_fens: List[str] = []
            node_draw_indices: List[int] = []
            board: chess.Board = node.state.board
            for idx, move in enumerate(node.legal_moves):
                board.push(move)
                node_fens.append(board.fen())
                if board.is_repetition() or board.is_fifty_moves() or board.is_stalemate():
                    node_draw_indices.append(idx)
                board.pop()
            return node_fens, node_draw_indices

        fens_and_draw_indices = [_get_fens_and_draw_indices(node) for node in nodes]
        fens, draw_indices = list(zip(*fens_and_draw_indices))
        fens = sum(fens, [])
        draw_indices = sum(draw_indices, [])

        return self._compute_evals_from_fens(fens, node_separations, draw_indices)

    def _compute_evals_from_fens(
        self,
        fens: List[str],
        node_separations: List[int],
        draw_indices: List[int],
    ) -> List[torch.Tensor]:
        """Compute win-prob evaluations for a list of FENs."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                white_evals = self.transformer.compute_white_win_prob_from_fen(fens, device=self.device)

                for idx in draw_indices:
                    white_evals[idx] = 0.5
                white_win_probs_split = torch.tensor_split(white_evals, node_separations)
        return list(white_win_probs_split)

    def _apply_root_prior_noise(self, priors: torch.Tensor) -> torch.Tensor:
        if priors.numel() == 0:
            return priors
        if self.root_dirichlet_eps <= 0.0 or self.root_dirichlet_alpha <= 0.0:
            return priors
        alpha = self.root_dirichlet_alpha
        eps = self.root_dirichlet_eps
        noise = torch.distributions.Dirichlet(
            torch.full_like(priors, alpha)
        ).sample()
        return (1.0 - eps) * priors + eps * noise

    def _temperatured_softmax(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        t = max(1e-6, float(temperature))
        return torch.softmax(logits / t, dim=0)

    def mcts(self, root: UCTNode, node_batch_size: int, time_limit: float = 2.0, verbose: bool = True) -> None:
        """Run MCTS from root for a time budget."""
        start_time = time.time()
        i = 0
        while True:
            nodes = []
            for _ in range(node_batch_size):
                node = root
                while node.is_expanded:
                    node = node.uct_select_child()
                nodes.append(node)
                node.add_virtual_loss()

            nodes_set = set(nodes)
            terminal_nodes = list(node for node in nodes_set if node.is_terminal())
            non_terminal_nodes = list(node for node in nodes_set if not node.is_terminal())

            for j, evals in enumerate(self.get_evals_for_all_moves(non_terminal_nodes)):
                node = non_terminal_nodes[j]

                priors = evals.clone()
                if node.state.board.turn == chess.BLACK:
                    priors = 1.0 - priors

                priors = self._temperatured_softmax(priors, self.prior_temperature)
                if node.is_root():
                    priors = self._apply_root_prior_noise(priors)

                node.expand(evals, child_priors=priors)
                node.backup(result=node.eval_based_on_children())

            for node in terminal_nodes:
                node.backup(result=node.terminal_state_eval())

            for node in nodes:
                node.revert_virtual_loss()

            i += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                break
        if verbose:
            print(f"Completed {i} iterations in {elapsed_time:.2f} seconds.")

    def _find_matching_root(self, game: ChessState) -> Optional[UCTNode]:
        if self._saved_root is None:
            return None
        target_fen = game.board.fen()
        if self._saved_root.state.board.fen() == target_fen:
            return self._saved_root

        # Check one ply (opponent's reply)
        for move_idx in range(len(self._saved_root.legal_moves)):
            child = self._saved_root.maybe_add_child(move_idx)
            if child.state.board.fen() == target_fen:
                return child

        return None

    def mcts_move(
        self,
        game: ChessState,
        node_batch_size: int = 1,
        time_limit: float = 2.0,
        verbose: bool = True,
    ) -> Tuple[chess.Move, float]:
        """Return best move and evaluation after running MCTS."""
        self._stop_ponder()
        root: Optional[UCTNode] = None
        saved_visits = 0.0
        if self.reuse_tree and self._saved_root is not None:
            matched = self._find_matching_root(game)
            if matched is None:
                print("MCTS reuse_tree: no matching node found (new game or game ended). Starting fresh root.")
            else:
                matched.parent = None
                matched.move = None
                matched.move_idx = None
                matched.plies_from_root = 0
                root = matched
                saved_visits = root.visits

        if root is None:
            root = UCTNode(device=self.device, exploration_weight=self.exploration_weight, state=game)

        with torch.no_grad():
            self.mcts(root, node_batch_size=node_batch_size, time_limit=time_limit, verbose=verbose)
        if verbose:
            print(root)

        if self.reuse_tree and saved_visits > 0:
            print(f"MCTS reuse_tree: saved {int(saved_visits)} visits by reusing the tree.")

        best_move_idx = root.best_move_idx()
        best_move = root.legal_moves[best_move_idx]
        eval = root.child_total_value[best_move_idx] / (1 + root.child_number_visits[best_move_idx])

        if self.reuse_tree:
            next_root = root.maybe_add_child(best_move_idx)
            next_root.parent = None
            next_root.move = None
            next_root.move_idx = None
            next_root.plies_from_root = 0
            self._saved_root = next_root
        else:
            self._saved_root = root

        if self.ponder and self._saved_root is not None and not self._saved_root.is_terminal():
            self._ponder_thread = threading.Thread(
                target=self._ponder_loop,
                args=(self._saved_root, node_batch_size),
                daemon=True,
            )
            self._ponder_thread.start()

        return best_move, eval.item()

    def get_evals(self, board: chess.Board, moves: List[chess.Move]) -> List[float]:
        """Return evaluations for candidate moves from the current player's perspective."""
        fen_list = []
        for move in moves:
            board.push(move)
            fen_list.append(board.fen())
            board.pop()

        with torch.no_grad():
            evals = self.transformer.compute_avg_bin_index_from_fens(fen_list, device=self.device)
            evals = evals / (self.transformer.bin_predictor.total_num_bins - 1)
            evals = evals if board.turn == chess.WHITE else 1.0 - evals
            return evals.detach().cpu().tolist()


def run() -> None:
    """Run a short demo game using MCTS."""
    set_seed(44)
    engine = MCTSEngine(
        config_path='configs/test_config.py',
    )
    board = chess.Board()
    state = ChessState(board)
    evals = [0.5]
    board.push(random.choice(list(board.legal_moves)))
    plies = 0
    print(board)
    while not state.is_terminal() and plies < 15:
        plies += 1
        move, eval = engine.mcts_move(state, node_batch_size=1, time_limit=10)
        print('Making move:', move, 'eval:', round(eval, 3), 'ply:', plies)
        state = state.make_move(move)
        print(state.board)
        if plies % 5 == 0:
            print(pgn.Game.from_board(state.board))
        if state.board.turn == chess.WHITE:
            eval = 1.0 - eval
        evals.append(eval)

    board = state.board

    print('Outcome:', board.outcome(claim_draw=True))

    game = pgn.Game().from_board(board)

    node = game
    for eval in evals:
        node = node.next()
        node.comment = f"[%wp {eval:.4f}]"

    pgn_string = game.accept(pgn.StringExporter(headers=True, variations=True, comments=True))

    print(pgn_string)
    print('Calls:', engine.calls_to_eval)


if __name__ == '__main__':
    run()
