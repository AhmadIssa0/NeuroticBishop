import argparse

from src.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, help="Path to a python config file with get_config()")
    parser.add_argument("--resume_training", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument(
        "--reset_optimizer",
        action="store_true",
        help="Skip loading optimizer state from checkpoint",
    )
    parser.add_argument(
        "--train_bin_proj_only",
        action="store_true",
        help="Only train ChessTransformer.proj_to_bin_predictor_logits",
    )
    args = parser.parse_args()

    train(
        config_path=args.config_path,
        resume_training=args.resume_training,
        reset_optimizer=args.reset_optimizer,
        train_bin_proj_only=args.train_bin_proj_only,
    )


if __name__ == "__main__":
    main()
