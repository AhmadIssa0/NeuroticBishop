from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    d_model: int = 512
    num_layers: int = 8
    nhead: int = 8
    dim_feedforward: Optional[int] = None
    norm_first: bool = True
    use_gelu: bool = False

    def resolved_dim_feedforward(self) -> int:
        return self.dim_feedforward if self.dim_feedforward is not None else 4 * self.d_model


@dataclass
class TrainerConfig:
    device: str = "cuda"
    dataset_path: str = "PATH_TO_YOUR_DATASET.jsonl"
    test_size: int = 25_000
    train_batch_size: int = 512
    eval_batch_size: int = 1024
    num_workers: int = 5
    prefetch_factor: int = 2
    seed: int = 42

    lr: float = 1e-4
    lr_after_resume: float = 5e-5
    max_norm: float = 0.3

    log_every: int = 100
    eval_every: int = 1000
    checkpoint_every: int = 500

    checkpoint_path: Optional[str] = "checkpoint_369500.pth"
    runs_dir: str = "runs/dModel_512_nlayers_8_nhead_8"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    # Optional metadata / experiment tracking
    name: str = "default"
    notes: Optional[str] = None