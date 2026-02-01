from src.configs.dataclasses import Config, TransformerConfig, TrainerConfig
from src.models.model import MixedCpMateWinProbMapper

def get_config(use_mixedcp_mapper=True):
    win_prob_mapper_fn = None
    if use_mixedcp_mapper:
        win_prob_mapper_fn = lambda bp: MixedCpMateWinProbMapper(
            bp,
            cp_scale=300.0,
            mate_gate_tau=0.40,
            use_mate_speed=True,
            mate_speed_kappa=0.003,
            overflow_center_margin_cp=300.0,
        )

    config = Config(
        model=TransformerConfig(
            d_model=512,
            num_layers=8,
            nhead=8,
            norm_first=True,
            use_gelu=True,
            win_prob_mapper_fn=win_prob_mapper_fn,
        ),
        trainer=TrainerConfig(
            experiment_dir=r"C:\Users\Ahmad-personal\PycharmProjects\chess_stockfish_evals_v2\experiments",
            device="cuda",
            dataset_path=r"C:\Users\Ahmad-personal\PycharmProjects\chess_stockfish_evals_v2\data\lichess_db_standard_rated_2025-11.jsonl",
            train_batch_size=512,
            eval_batch_size=512,
            num_workers=2,
            prefetch_factor=5,
            keep_checkpoint_iters=[1_245_000],
        ),
        exp_name="old_512_transformer",
        notes="old pretty stable run with different BinPredictor, continued training with new BinPredictor",
    )
    return config

