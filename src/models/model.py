import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence, Tuple, Optional

from src.dataset.lichess_dataset import FEN_CHAR_TO_INDEX, MAX_FEN_LENGTH
from src.dataset.fen_utils import expand_fen_string, remove_full_half_moves_from_fen
from src.bin_predictor import BinPredictor


# -----------------------------
# Win-prob mappers (plug-ins)
# -----------------------------

class WinProbMapper(nn.Module):
    """
    Interface: map a (B, T) bin probability distribution to a (B,) white win probability in [0, 1].
    """
    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AvgBinIndexWinProbMapper(WinProbMapper):
    """
    Your current behavior: E[idx] / (T-1).
    This is mostly here as a drop-in default mapper.
    """
    def __init__(self, total_num_bins: int):
        super().__init__()
        self.total_num_bins = int(total_num_bins)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        # probs: (B, T)
        B, T = probs.shape
        assert T == self.total_num_bins
        idx = torch.arange(T, device=probs.device, dtype=probs.dtype).view(1, -1)
        avg = (probs * idx).sum(dim=1)
        denom = max(1, T - 1)
        return torch.clamp(avg / float(denom), 0.0, 1.0)


class MixedCpMateWinProbMapper(WinProbMapper):
    """
    Concept:
      - Split distribution into mate-for-us bins, mate-against-us bins, and eval (cp) bins.
      - Convert cp bins to win probability with sigmoid(cp / cp_scale).
      - Convert mate bins to near-terminal value in [-1,1] based on mate mass direction.
      - Gate between cp and mate using total mate probability p_mate and threshold mate_gate_tau.
      - (Optional) tiny mate-speed tie-break (recommended OFF for MCTS backup; useful only at root ordering).
    """
    def __init__(
        self,
        bin_predictor: BinPredictor,
        cp_scale: float = 300.0,
        mate_gate_tau: float = 0.40,
        use_mate_speed: bool = False,
        mate_speed_kappa: float = 0.003,
        overflow_center_margin_cp: float = 300.0,
    ):
        super().__init__()
        self.bp = bin_predictor
        self.cp_scale = float(cp_scale)
        self.mate_gate_tau = float(mate_gate_tau)
        self.use_mate_speed = bool(use_mate_speed)
        self.mate_speed_kappa = float(mate_speed_kappa)

        # Precompute representative cp values for eval bins in the exact order they appear in indices.
        # Eval bin indices are: [m+1 .. T-m-2] inclusive
        # That corresponds to: cp < ticks[0], intervals between ticks, cp >= ticks[-1]
        ticks = torch.tensor(self.bp.bin_ticks, dtype=torch.float32)

        under = ticks[0] - float(overflow_center_margin_cp)
        mids = 0.5 * (ticks[:-1] + ticks[1:])
        over = ticks[-1] + float(overflow_center_margin_cp)
        centers = torch.cat([under.view(1), mids, over.view(1)], dim=0)

        self.register_buffer("eval_bin_centers_cp", centers, persistent=False)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """
        probs: (B, T) softmax probabilities over all bins.
        returns: (B,) white win probability in [0, 1]
        """
        probs = probs.to(dtype=torch.float32)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

        B, T = probs.shape
        m = self.bp.max_mate_range

        # --- Slice regions (matches BinPredictor indexing) ---
        neg_mate = probs[:, 0:m]                  # mate -1..-m
        neg_over = probs[:, m:m+1]                # mate <= -(m+1)

        eval_start = m + 1
        eval_end = T - m - 2
        eval_probs = probs[:, eval_start:eval_end+1]  # cp underflow + mids + overflow

        pos_over = probs[:, T - m - 1:T - m]      # mate >= +(m+1) (shape (B,1))
        pos_mate = probs[:, T - m:T]              # mate +m..+1

        # --- Mate masses ---
        p_mate_minus = neg_mate.sum(dim=1) + neg_over.squeeze(1)
        p_mate_plus  = pos_mate.sum(dim=1) + pos_over.squeeze(1)
        p_mate = p_mate_minus + p_mate_plus

        # --- CP -> win prob (conditional on being in eval bins) ---
        centers = self.eval_bin_centers_cp.to(device=probs.device, dtype=probs.dtype)
        # safety: ensure centers length matches eval_probs bins
        if centers.numel() != eval_probs.size(1):
            raise RuntimeError(
                f"eval_bin_centers_cp has {centers.numel()} entries but eval_probs has {eval_probs.size(1)} bins. "
                "BinPredictor layout mismatch."
            )

        x = (centers / self.cp_scale).clamp(-20.0, 20.0)
        pwin_per_eval_bin = torch.sigmoid(x).view(1, -1)  # (1, N_eval_bins)

        p_eval_sum = eval_probs.sum(dim=1).clamp_min(1e-12)
        eval_cond = eval_probs / p_eval_sum.unsqueeze(1)
        Pwin_cp = (eval_cond * pwin_per_eval_bin).sum(dim=1)  # (B,)
        v_cp = 2.0 * Pwin_cp - 1.0                             # [-1,1]

        # --- Mate -> terminal-ish value in [-1,1] ---
        v_mate = (p_mate_plus - p_mate_minus) / p_mate.clamp_min(1e-12)
        v_mate = torch.clamp(v_mate, -1.0, 1.0)

        # --- Gate: when mate dominates ---
        tau = self.mate_gate_tau
        alpha = (p_mate - tau) / (1.0 - tau)
        alpha = torch.clamp(alpha, 0.0, 1.0)

        v = (1.0 - alpha) * v_cp + alpha * v_mate
        v = torch.clamp(v, -1.0, 1.0)

        # --- Optional tiny mate-speed tie-break (generally root-only) ---
        if self.use_mate_speed and self.mate_speed_kappa > 0.0:
            # Expected mate distances conditional on + or - mate
            # Negative mates: distances 1..m, overflow treated as m+1
            d_neg = torch.arange(1, m + 1, device=probs.device, dtype=probs.dtype).view(1, -1)
            p_neg = p_mate_minus.clamp_min(1e-12)
            E_d_neg = ((neg_mate * d_neg).sum(dim=1) + neg_over.squeeze(1) * (m + 1.0)) / p_neg

            # Positive mates: bins correspond to mate +m..+1, so distances m..1
            d_pos = torch.arange(m, 0, -1, device=probs.device, dtype=probs.dtype).view(1, -1)
            p_pos = p_mate_plus.clamp_min(1e-12)
            E_d_pos = ((pos_mate * d_pos).sum(dim=1) + pos_over.squeeze(1) * (m + 1.0)) / p_pos

            frac_pos = p_mate_plus / p_mate.clamp_min(1e-12)
            frac_neg = p_mate_minus / p_mate.clamp_min(1e-12)

            norm = float(m)
            t = alpha * self.mate_speed_kappa * (
                frac_pos * (-(E_d_pos / norm)) +   # prefer smaller mate distance for us
                frac_neg * (+(E_d_neg / norm))     # prefer larger mate distance against us (delay)
            )
            v = torch.clamp(v + t, -1.0, 1.0)

        # Convert value [-1,1] -> win prob [0,1]
        Pwin = 0.5 * (v + 1.0)
        return torch.clamp(Pwin, 0.0, 1.0)


# -----------------------------
# ChessTransformer (minimal changes)
# -----------------------------

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
        # NEW (minimal): optional mapper object
        win_prob_mapper: Optional[WinProbMapper] = None,
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

        # NEW (minimal): if not provided, keep your old behavior via default mapper
        self.win_prob_mapper = win_prob_mapper or AvgBinIndexWinProbMapper(bin_predictor.total_num_bins)

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

    def compute_white_win_prob_from_fen(self, fen_list: Sequence[str], device: str) -> torch.Tensor:
        emb_indices = self.fen_list_to_emb_indices(fen_list, device)
        return self.compute_white_win_prob(emb_indices)

    def compute_white_win_prob(self, emb_indices: torch.LongTensor) -> torch.Tensor:
        # MINIMAL CHANGE: use mapper over full probability vector
        probs = self.compute_bin_probabilities(emb_indices)
        return self.win_prob_mapper(probs)

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
