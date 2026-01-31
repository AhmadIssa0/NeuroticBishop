from src.configs.dataclasses import Config, TransformerConfig, TrainerConfig


def get_config():
    config = Config(
        model=TransformerConfig(
            d_model=512,
            num_layers=8,
            nhead=8,
            norm_first=True,
            use_gelu=False,
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
        exp_name="test_run",
        notes="baseline run",
    )
    return config

