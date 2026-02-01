from dataclasses import dataclass, field
from typing import Optional, List, Callable
from abc import ABC, abstractmethod
from src.bin_predictor import BinPredictor
from src.models.model import ChessTransformer
from src.models.relative_position_model import ChessRelativeTransformer
from src.models.model import WinProbMapper


@dataclass
class ModelConfig(ABC):
    """Abstract base config for model construction."""

    @abstractmethod
    def create_model(self, bin_predictor: BinPredictor, device: str):
        """Create and return a model instance placed on the given device."""
        raise NotImplementedError


@dataclass
class TransformerConfig(ModelConfig):
    d_model: int = 512
    num_layers: int = 8
    nhead: int = 8
    dim_feedforward: Optional[int] = None
    norm_first: bool = True
    use_gelu: bool = False
    win_prob_mapper_fn: Optional[Callable[[BinPredictor], WinProbMapper]] = None

    def resolved_dim_feedforward(self) -> int:
        return self.dim_feedforward if self.dim_feedforward is not None else 4 * self.d_model

    def create_model(self, bin_predictor: BinPredictor, device: str):
        win_prob_mapper = self.win_prob_mapper_fn(bin_predictor) if self.win_prob_mapper_fn else None
        return ChessTransformer(
            bin_predictor=bin_predictor,
            d_model=self.d_model,
            num_layers=self.num_layers,
            nhead=self.nhead,
            dim_feedforward=self.resolved_dim_feedforward(),
            norm_first=self.norm_first,
            use_gelu=self.use_gelu,
            win_prob_mapper=win_prob_mapper,
        ).to(device=device)


@dataclass
class RelativeTransformerConfig(ModelConfig):
    d_model: int = 512
    nhead: int = 8
    board_num_layers: int = 8
    fusion_num_layers: int = 2
    dim_feedforward: Optional[int] = None
    norm_first: bool = True
    use_gelu: bool = False
    dropout_p: float = 0.0
    use_square_abs_embedding: bool = True

    def resolved_dim_feedforward(self) -> int:
        return self.dim_feedforward if self.dim_feedforward is not None else 4 * self.d_model

    def create_model(self, bin_predictor: BinPredictor, device: str):
        return ChessRelativeTransformer(
            bin_predictor=bin_predictor,
            d_model=self.d_model,
            nhead=self.nhead,
            board_num_layers=self.board_num_layers,
            fusion_num_layers=self.fusion_num_layers,
            dim_feedforward=self.resolved_dim_feedforward(),
            norm_first=self.norm_first,
            use_gelu=self.use_gelu,
            dropout_p=self.dropout_p,
            use_square_abs_embedding=self.use_square_abs_embedding,
        ).to(device=device)


@dataclass
class TrainerConfig:
    device: str = "cuda"
    dataset_path: str = r"C:\Users\Ahmad-personal\PycharmProjects\chess_stackfish_evals\data\lichess_db_standard_rated_2024-02.jsonl"
    test_size: int = 5_000
    train_batch_size: int = 512
    eval_batch_size: int = 1024
    num_workers: int = 5
    prefetch_factor: int = 2
    seed: int = None

    lr: float = 1e-4
    max_norm: float = 0.3

    log_every: int = 100
    eval_every: int = 1000
    checkpoint_every: int = 500
    keep_checkpoint_iters: List[int] = field(default_factory=list)

    experiment_dir: str = "experiments"
    checkpoint_path: Optional[str] = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=TransformerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    # Optional metadata / experiment tracking
    exp_name: str = "default"
    notes: Optional[str] = None
