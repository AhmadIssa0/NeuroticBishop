from src.configs.dataclasses import Config, RelativeTransformerConfig, TrainerConfig


def get_config():
    config = Config(
        model=RelativeTransformerConfig(
            d_model=512,
            nhead=8,
            board_num_layers=8,
            fusion_num_layers=2,
            norm_first=True,
            use_gelu=False,
            dropout_p=0.0,
            use_square_abs_embedding=True,
        ),
        trainer=TrainerConfig(
            experiment_dir=r"C:\Users\Ahmad-personal\PycharmProjects\chess_stockfish_evals_v2\experiments",
            device="cuda",
            dataset_path=r"C:\Users\Ahmad-personal\PycharmProjects\chess_stockfish_evals_v2\data\lichess_db_standard_rated_2024-02.jsonl",
            train_batch_size=256,
            eval_batch_size=512,
            num_workers=1,
            prefetch_factor=2,
        ),
        exp_name="test_relative_run",
        notes="relative transformer baseline",
    )
    return config