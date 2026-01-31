from src.configs.dataclasses import Config, TransformerConfig, TrainerConfig


def get_config():
    config = Config(
        model=TransformerConfig(
            d_model=512,
            num_layers=8,
            nhead=8,
            norm_first=True,
            use_gelu=True,
        ),
        trainer=TrainerConfig(
            experiment_dir=r"C:\Users\Ahmad-personal\PycharmProjects\chess_stockfish_evals_v2\experiments",
            device="cuda",
            dataset_path=r"C:\Users\Ahmad-personal\PycharmProjects\chess_stockfish_evals_v2\data\lichess_db_standard_rated_2024-02.jsonl",
            train_batch_size=256,
            eval_batch_size=512,
            num_workers=1,
            prefetch_factor=2,
            keep_checkpoint_iters=[1_245_000],
        ),
        exp_name="old_512_transformer",
        notes="old pretty stable run with different BinPredictor, continued training with new BinPredictor",
    )
    return config

