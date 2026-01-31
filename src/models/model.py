import torch
import torch.nn as nn
from typing import List, Sequence, Tuple
from src.dataset.lichess_dataset import FEN_CHAR_TO_INDEX, MAX_FEN_LENGTH
from src.dataset.fen_utils import expand_fen_string, remove_full_half_moves_from_fen
from src.bin_predictor import BinPredictor
import torch.nn.functional as F


class ChessTransformer(nn.Module):

    def __init__(
        self,
        bin_predictor: BinPredictor,
        d_model: int = 512,
        num_layers: int = 8,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        norm_first: bool = True,
        use_gelu: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=len(FEN_CHAR_TO_INDEX), embedding_dim=d_model)
        # Note that this next output head isn't being used! Remove it later...
        self.proj_to_win_prob_logit = nn.Linear(d_model, 1)
        activation_type = "gelu" if use_gelu else "relu"
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=norm_first,
            activation=activation_type,
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)

        # Use a learned positional embedding
        self.pos_embedding = nn.Embedding(num_embeddings=MAX_FEN_LENGTH + 1, embedding_dim=d_model)

        self.proj_to_bin_predictor_logits = nn.Linear(d_model, bin_predictor.total_num_bins)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.bin_predictor = bin_predictor

    def compute_cross_entropy_loss(
        self,
        emb_indices: torch.LongTensor,
        bin_pred_classes: torch.LongTensor,
    ) -> torch.Tensor:
        logits = self.compute_bin_predictor_logits(emb_indices)
        return self.cross_entropy_loss(logits, bin_pred_classes)

    def compute_cross_entropy_loss_from_probs(
        self,
        emb_indices: torch.LongTensor,
        bin_class_probs: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        logits = self.compute_bin_predictor_logits(emb_indices)
        return self.cross_entropy_loss(logits / temperature, bin_class_probs)

    def compute_bin_predictor_logits(self, emb_indices: torch.LongTensor) -> torch.Tensor:
        embedding = self.embedding(emb_indices)  # (B, MAX_FEN_LENGTH + 1, d_model)
        embedding = embedding + self.pos_embedding.weight.unsqueeze(0)
        output = self.transformer(embedding)
        # Only look at the first token's output (i.e. eval token)
        return self.proj_to_bin_predictor_logits(self.layer_norm(output[:, 0, :]))

    def compute_bin_probabilities(self, emb_indices: torch.LongTensor, temperature: float = 1.0) -> torch.Tensor:
        return (self.compute_bin_predictor_logits(emb_indices) / temperature).softmax(dim=1)

    def compute_bin_probabilities_from_fens(self, fen_list: Sequence[str], device: str) -> torch.Tensor:
        return self.forward(self.fen_list_to_emb_indices(fen_list, device))

    def compute_avg_bin_index(self, emb_indices: torch.LongTensor) -> torch.Tensor:
        class_probs = self.compute_bin_probabilities(emb_indices)
        class_indices = torch.arange(0, self.bin_predictor.total_num_bins, device=emb_indices.device).unsqueeze(0)
        return (class_probs * class_indices).sum(dim=1)

    def compute_bin_index_means_and_stds_from_fens(
        self,
        fen_list: Sequence[str],
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probabilities = self.compute_bin_probabilities_from_fens(fen_list, device)
        indices = torch.arange(0, self.bin_predictor.total_num_bins, device=device).unsqueeze(0)

        mean_indices = torch.sum(indices * probabilities, dim=1)
        squared_diffs = (indices - mean_indices.unsqueeze(1)) ** 2
        variances = torch.sum(squared_diffs * probabilities, dim=1)
        std_deviations = torch.sqrt(variances)
        return mean_indices, std_deviations

    def forward(self, emb_indices: torch.LongTensor, temperature: float = 1.0) -> torch.Tensor:
        return (self.compute_bin_predictor_logits(emb_indices) / temperature).softmax(dim=1)

    def compute_avg_bin_index_from_fens(self, fen_list: Sequence[str], device: str) -> torch.Tensor:
        return self.compute_avg_bin_index(self.fen_list_to_emb_indices(fen_list, device))

    def fen_list_to_emb_indices(self, fen_list: Sequence[str], device: str) -> torch.LongTensor:
        emb_indices_batch: List[torch.LongTensor] = []
        for fen_str in fen_list:
            expanded_fen = remove_full_half_moves_from_fen(expand_fen_string(fen_str))
            indices_lst = [FEN_CHAR_TO_INDEX['eval']] + [FEN_CHAR_TO_INDEX[c] for c in expanded_fen]
            indices_lst += [FEN_CHAR_TO_INDEX[' '] for _ in range(MAX_FEN_LENGTH - len(expanded_fen))]
            emb_indices_batch.append(torch.tensor(indices_lst, dtype=torch.long))
        return torch.stack(emb_indices_batch, dim=0).to(device=device)

    # def compute_white_win_prob_from_fen(self, fen_list: Sequence[str], device: str) -> torch.Tensor:
    #     emb_indices = self.fen_list_to_emb_indices(fen_list, device)
    #     return self.compute_white_win_prob(emb_indices)
    #
    # def compute_white_win_prob(self, emb_indices: torch.LongTensor) -> torch.Tensor:
    #     embedding = self.embedding(emb_indices)  # (B, MAX_FEN_LENGTH + 1, d_model)
    #     embedding = embedding + self.pos_embedding.weight.unsqueeze(0)
    #     output = self.transformer(embedding)
    #     return self.proj_to_win_prob_logit(self.layer_norm(output[:, 0, :])).squeeze(-1).sigmoid()

    def compute_white_win_prob_from_fen(self, fen_list: Sequence[str], device: str) -> torch.Tensor:
        emb_indices = self.fen_list_to_emb_indices(fen_list, device)
        return self.compute_white_win_prob(emb_indices)

    def compute_white_win_prob(self, emb_indices: torch.LongTensor) -> torch.Tensor:
        avg_bin_index = self.compute_avg_bin_index(emb_indices)
        denom = max(1, self.bin_predictor.total_num_bins - 1)
        return avg_bin_index / denom

    def _get_soft_target_sigma(self, num_classes: int, target_center_prob: float) -> float:
        cache_key = (num_classes, float(target_center_prob))
        if not hasattr(self, "_soft_target_sigma_cache"):
            self._soft_target_sigma_cache = {}
        if cache_key in self._soft_target_sigma_cache:
            return self._soft_target_sigma_cache[cache_key]

        # Binary search for sigma so that center prob ~= target_center_prob
        center = num_classes // 2
        distances = torch.arange(num_classes, dtype=torch.float32) - center
        distances_sq = distances ** 2

        low, high = 1e-3, 1000.0
        for _ in range(40):
            mid = (low + high) / 2.0
            logits = -distances_sq / (2.0 * mid * mid)
            probs = torch.softmax(logits, dim=0)
            center_prob = probs[center].item()
            if center_prob > target_center_prob:
                low = mid
            else:
                high = mid

        sigma = (low + high) / 2.0
        self._soft_target_sigma_cache[cache_key] = sigma
        return sigma

    def compute_soft_cross_entropy_loss(
        self,
        emb_indices: torch.LongTensor,
        bin_pred_classes: torch.LongTensor,
        target_center_prob: float = 0.8,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        logits = self.compute_bin_predictor_logits(emb_indices) / temperature
        num_classes = logits.size(1)
        sigma = self._get_soft_target_sigma(num_classes, target_center_prob)

        class_indices = torch.arange(num_classes, device=logits.device).unsqueeze(0)
        distances_sq = (class_indices - bin_pred_classes.unsqueeze(1)).float().pow(2)
        target_logits = -distances_sq / (2.0 * sigma * sigma)
        target_probs = torch.softmax(target_logits, dim=1)

        log_probs = torch.log_softmax(logits, dim=1)
        return -(target_probs * log_probs).sum(dim=1)