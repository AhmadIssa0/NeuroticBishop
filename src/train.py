import glob
import os
import random
from functools import partial
from typing import Tuple, Optional
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter

from src.bin_predictor import BinPredictor
from src.dataset.lichess_dataset import JSONLinesLichessDataset, collate_fn
from src.models.model import ChessTransformer


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


def setup_optimizer_and_scaler(transformer: ChessTransformer, lr: float) -> Tuple[Optimizer, GradScaler]:
    scaler = torch.cuda.amp.GradScaler()
    optim = Adam(transformer.parameters(), lr=lr)
    return optim, scaler


def load_checkpoint_if_available(
    checkpoint_path: str,
    transformer: ChessTransformer,
    scaler: GradScaler,
    optim: Optimizer,
    reset_optimizer: bool,
) -> int:
    global_step = 0
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint from:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        transformer.load_state_dict(checkpoint['transformer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if not reset_optimizer:
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
    return global_step


def adjust_learning_rate(optim: Optimizer, lr: float) -> None:
    for param_group in optim.param_groups:
        param_group['lr'] = lr


def _extract_checkpoint_step(path: str) -> int:
    try:
        return int(Path(path).stem.split("_")[-1])
    except ValueError:
        return -1


def save_checkpoint(
    transformer: ChessTransformer,
    scaler: GradScaler,
    optim: Optimizer,
    global_step: int,
    checkpoint_dir: str,
    keep_checkpoint_iters: set[int],
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    for checkpoint_file in glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth")):
        step = _extract_checkpoint_step(checkpoint_file)
        if step in keep_checkpoint_iters:
            continue
        os.remove(checkpoint_file)
        print(f'Deleted {checkpoint_file}')

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{global_step}.pth')
    torch.save(
        {
            'transformer_state_dict': transformer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'global_step': global_step,
        },
        checkpoint_path
    )
    print(f'Saved model at {checkpoint_path}.')


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_prob_of_win(transformer: ChessTransformer, dataloader: DataLoader, dataset_name: str,
                         loss_fn: torch.nn.Module, global_step: int, summary_writer: SummaryWriter,
                         device: str) -> None:
    with torch.no_grad():
        total_test_samples = 0
        acc_loss = 0.0
        for test_batch in dataloader:
            test_batch = test_batch.to(device=device)
            pred_white_win_prob = transformer.compute_white_win_prob(test_batch.embedding_indices)
            total_test_samples += len(test_batch.cp_valid)
            acc_loss += loss_fn(pred_white_win_prob, test_batch.white_win_prob).sum().item()

        loss = acc_loss / total_test_samples
        print(f'{dataset_name} loss: {loss}, Total samples: {total_test_samples}')
        summary_writer.add_scalar(f'Loss/{dataset_name}', loss, global_step)

        print('Pred_win_prob:', pred_white_win_prob[:10])
        print('True_win_prob:', test_batch.white_win_prob[:10])
        print('Valid?', test_batch.cp_valid[:10])


def evaluate_cross_entropy(transformer: ChessTransformer, dataloader: DataLoader, dataset_name: str, global_step: int,
                           summary_writer: SummaryWriter, device: str) -> None:
    with torch.no_grad():
        total_test_samples = 0
        acc_loss = 0.0
        for test_batch in dataloader:
            test_batch = test_batch.to(device=device)
            total_test_samples += len(test_batch.cp_valid)
            acc_loss += transformer.compute_soft_cross_entropy_loss(
                emb_indices=test_batch.embedding_indices,
                bin_pred_classes=test_batch.bin_classes,
                target_center_prob=0.8
            ).sum().item()

        loss = acc_loss / total_test_samples
        print(f'{dataset_name} loss: {loss}, Total samples: {total_test_samples}')
        summary_writer.add_scalar(f'CrossEntropyLoss/{dataset_name}', loss, global_step)


def build_dataloaders(
    dataset_path: str,
    test_size: int,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    seed: int,
    eval_num_workers: int | None = None,
    eval_prefetch_factor: int | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, BinPredictor]:
    dataset = JSONLinesLichessDataset(dataset_path)
    print('Dataset size:', len(dataset))
    print(dataset.__getitem__(11000))

    if seed is not None:
        set_seed(seed)

    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    subset_indices = torch.randperm(len(train_dataset))[:test_size]
    subset_dataset = Subset(train_dataset, subset_indices)

    bin_pred = BinPredictor()
    batched_collate_fn = partial(collate_fn, bin_predictor=bin_pred)

    eval_num_workers = eval_num_workers if eval_num_workers is not None else max(0, num_workers // 2)
    eval_prefetch_factor = eval_prefetch_factor if eval_prefetch_factor is not None else 1

    train_eval_kwargs = {
        "batch_size": eval_batch_size,
        "shuffle": False,
        "num_workers": eval_num_workers,
        "collate_fn": batched_collate_fn,
        "pin_memory": False,
        "drop_last": False,
        "persistent_workers": False,
    }
    if eval_num_workers > 0:
        train_eval_kwargs["prefetch_factor"] = eval_prefetch_factor

    train_kwargs = {
        "batch_size": train_batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "collate_fn": batched_collate_fn,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": True,
    }
    if num_workers > 0:
        train_kwargs["prefetch_factor"] = prefetch_factor

    test_kwargs = {
        "batch_size": eval_batch_size,
        "shuffle": False,
        "num_workers": eval_num_workers,
        "collate_fn": batched_collate_fn,
        "pin_memory": False,
        "drop_last": False,
        "persistent_workers": False,
    }
    if eval_num_workers > 0:
        test_kwargs["prefetch_factor"] = eval_prefetch_factor

    train_eval_dataloader = DataLoader(subset_dataset, **train_eval_kwargs)
    train_dataloader = DataLoader(train_dataset, **train_kwargs)
    test_dataloader = DataLoader(test_dataset, **test_kwargs)

    return train_dataloader, train_eval_dataloader, test_dataloader, bin_pred


def train_step(transformer: ChessTransformer, batch, loss_fn: torch.nn.Module, device: str) -> torch.Tensor:
    with autocast():
        batch = batch.to(device=device)
        loss = transformer.compute_soft_cross_entropy_loss(
            emb_indices=batch.embedding_indices,
            bin_pred_classes=batch.bin_classes,
            target_center_prob=0.8
        ).mean()
    return loss


def log_training_metrics(summary_writer: SummaryWriter, global_step: int, grad_norm: torch.Tensor,
                         loss: torch.Tensor) -> None:
    print('Global step:', global_step)
    summary_writer.add_scalar('Loss/GradientNorm', grad_norm, global_step)
    summary_writer.add_scalar('Loss/TrainLoss', loss, global_step)


def run_eval_if_needed(transformer: ChessTransformer, test_dataloader: DataLoader, train_eval_dataloader: DataLoader,
                       loss_fn: torch.nn.Module, global_step: int, summary_writer: SummaryWriter, device: str,
                       eval_every: int) -> None:
    if global_step % eval_every != 0:
        return
    transformer.eval()
    evaluate_cross_entropy(transformer, test_dataloader, 'Test-Set', global_step, summary_writer, device)
    evaluate_cross_entropy(transformer, train_eval_dataloader, 'Train-Eval-Set', global_step, summary_writer, device)
    transformer.train()


def train(
        *,
        config_path: str,
        resume_training: bool = False,
        reset_optimizer: bool = False,
        train_bin_proj_only: bool = False,
) -> None:
    config = load_config_from_path(config_path)

    trainer = config.trainer
    model_cfg = config.model

    experiment_root = Path(trainer.experiment_dir) / config.exp_name
    checkpoint_dir = experiment_root / "checkpoints"
    summary_dir = experiment_root / "summaries"

    checkpoint_path = trainer.checkpoint_path
    if resume_training:
        latest = find_latest_checkpoint(checkpoint_dir)
        if latest is not None:
            checkpoint_path = latest

    train_dataloader, train_eval_dataloader, test_dataloader, bin_pred = build_dataloaders(
        dataset_path=trainer.dataset_path,
        test_size=trainer.test_size,
        train_batch_size=trainer.train_batch_size,
        eval_batch_size=trainer.eval_batch_size,
        num_workers=trainer.num_workers,
        prefetch_factor=trainer.prefetch_factor,
        seed=trainer.seed
    )

    transformer = model_cfg.create_model(bin_predictor=bin_pred, device=trainer.device)
    if train_bin_proj_only:
        for param in transformer.parameters():
            param.requires_grad = False
        for param in transformer.proj_to_bin_predictor_logits.parameters():
            param.requires_grad = True
        optim = Adam(transformer.proj_to_bin_predictor_logits.parameters(), lr=trainer.lr)
        scaler = torch.cuda.amp.GradScaler()
    else:
        optim, scaler = setup_optimizer_and_scaler(transformer, lr=trainer.lr)

    if checkpoint_path is not None:
        global_step = load_checkpoint_if_available(
            checkpoint_path,
            transformer,
            scaler,
            optim,
            reset_optimizer=reset_optimizer,
        )
    else:
        global_step = 0

    summary_writer = SummaryWriter(str(summary_dir))
    keep_checkpoint_iters = set(trainer.keep_checkpoint_iters)
    # l2_loss_fn = torch.nn.MSELoss(reduction='none')

    for batch in train_dataloader:
        global_step += 1
        optim.zero_grad()

        if global_step == 1:
            with torch.no_grad():
                batch = batch.to(device=trainer.device)
                num_classes = transformer.bin_predictor.total_num_bins
                sigma = transformer._get_soft_target_sigma(num_classes, 1.0)

                class_indices = torch.arange(num_classes, device=trainer.device).unsqueeze(0)
                distances_sq = (class_indices - batch.bin_classes.unsqueeze(1)).float().pow(2)
                target_logits = -distances_sq / (2.0 * sigma * sigma)
                target_probs = torch.softmax(target_logits, dim=1)

                print("Soft target distribution examples (first batch):")
                for i in range(min(2, target_probs.size(0))):
                    top_vals, top_idx = torch.topk(target_probs[i], k=8)
                    print(f"  sample {i} | true_bin={batch.bin_classes[i].item()} "
                          f"| top bins={top_idx.tolist()} | probs={top_vals.tolist()}")

        loss = train_step(transformer, batch, None, trainer.device)

        scaler.scale(loss).backward()

        scaler.unscale_(optim)
        grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), trainer.max_norm)

        scaler.step(optim)
        scaler.update()

        if global_step % trainer.log_every == 0:
            log_training_metrics(summary_writer, global_step, grad_norm, loss)

        run_eval_if_needed(
            transformer=transformer,
            test_dataloader=test_dataloader,
            train_eval_dataloader=train_eval_dataloader,
            loss_fn=None,
            global_step=global_step,
            summary_writer=summary_writer,
            device=trainer.device,
            eval_every=trainer.eval_every,
        )

        if global_step % trainer.checkpoint_every == 0 or global_step in keep_checkpoint_iters:
            save_checkpoint(
                transformer,
                scaler,
                optim,
                global_step,
                str(checkpoint_dir),
                keep_checkpoint_iters=keep_checkpoint_iters,
            )

