# chess_transformer_2d_relative.py
#
# Drop-in replacement for your current ChessTransformer, but:
# - Uses 64 board-square tokens (8x8) + 1 eval token (total length 65)
# - Canonicalizes to "side-to-move" viewpoint by flipping the board when stm == 'b'
# - Encodes non-board FEN info (castling rights + en passant) as embeddings ADDED to square embeddings
# - Uses Shaw-style 2D relative self-attention on the 64 board tokens
# - Optionally runs a small standard Transformer "fusion" stack so the eval token can aggregate board info
#
# Public methods/structure match your current class (same method names and signatures).
# (Init args differ; that's fine per your note.)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bin_predictor import BinPredictor


# =========================
# FEN parsing / canonicalization
# =========================

_PIECE_TO_INDEX: Dict[str, int] = {
    ".": 0,
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}
_INDEX_TO_PIECE: Dict[int, str] = {v: k for k, v in _PIECE_TO_INDEX.items()}

BOARD_VOCAB_SIZE = 13  # ".", PNBRQKpnbrqk


def _expand_board(board_fen: str) -> str:
    # board_fen is like: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    out = []
    for ch in board_fen:
        if ch == "/":
            continue
        if ch.isdigit():
            out.append("." * int(ch))
        else:
            out.append(ch)
    s = "".join(out)
    if len(s) != 64:
        raise ValueError(f"Expanded board should be 64 chars, got {len(s)} from {board_fen!r}")
    return s


def _square_to_xy(square: str) -> Tuple[int, int]:
    # "a1" -> (0,0), "h8" -> (7,7)
    file = ord(square[0]) - ord("a")
    rank = ord(square[1]) - ord("1")
    if not (0 <= file <= 7 and 0 <= rank <= 7):
        raise ValueError(f"Bad square: {square!r}")
    return file, rank


def _xy_to_square(file: int, rank: int) -> str:
    return chr(ord("a") + file) + chr(ord("1") + rank)


def _rotate_180_square(square: str) -> str:
    # 180-degree rotation: a1 <-> h8, a8 <-> h1, etc.
    f, r = _square_to_xy(square)       # file 0..7, rank 0..7
    f2, r2 = 7 - f, 7 - r
    return _xy_to_square(f2, r2)


def _board_index_a8_to_h1(file: int, rank: int) -> int:
    """
    Convert (file, rank) where rank 0 is '1' into index with ordering a8..h1:
    - index 0  = a8
    - index 7  = h8
    - index 56 = a1
    - index 63 = h1
    """
    # rank: 0..7 for ranks 1..8
    row_from_top = 7 - rank
    return row_from_top * 8 + file


def _parse_fen_parts(fen: str) -> Tuple[str, str, str, str]:
    """
    Returns: (expanded_board_64, stm, castling, ep)
    Ignores halfmove/fullmove counters if present.
    """
    parts = fen.strip().split()
    if len(parts) < 4:
        raise ValueError(f"FEN must have at least 4 fields, got: {fen!r}")
    board_fen, stm, castling, ep = parts[0], parts[1], parts[2], parts[3]
    expanded = _expand_board(board_fen)
    if stm not in ("w", "b"):
        raise ValueError(f"Bad side-to-move in FEN: {stm!r}")
    if castling == "":
        castling = "-"
    if ep == "":
        ep = "-"
    return expanded, stm, castling, ep


def _canonicalize_to_stm_view(
    expanded_board_64: str,
    stm: str,
    castling: str,
    ep: str,
) -> Tuple[str, int, int, int, int, int]:
    """
    Canonicalize to "side-to-move is white" viewpoint:
      - if stm == 'w': keep board as-is
      - if stm == 'b': rotate board 180 degrees AND swap colors (case)
    Also returns castling-right bits in canonical frame:
      myK, myQ, oppK, oppQ as ints in {0,1}
    And en passant index in canonical frame:
      ep_index in [0..63] or -1 if none
    """
    if stm == "w":
        board_can = expanded_board_64
        # In canonical frame, "my" is White, "opp" is Black
        myK = 1 if "K" in castling else 0
        myQ = 1 if "Q" in castling else 0
        oppK = 1 if "k" in castling else 0
        oppQ = 1 if "q" in castling else 0

        if ep == "-":
            ep_idx = -1
        else:
            # ep like "e3"
            f, r = _square_to_xy(ep)
            ep_idx = _board_index_a8_to_h1(f, r)
        return board_can, myK, myQ, oppK, oppQ, ep_idx

    # stm == 'b': rotate + swap case so that side-to-move becomes "white" in canonical frame
    # Rotate in a8..h1 indexing: 180-degree rotation corresponds to reversing the 64 list.
    # Then swap case to swap colors.
    rev = expanded_board_64[::-1]
    board_can = "".join(ch.swapcase() if ch.isalpha() else ch for ch in rev)

    # Castling rights swap because colors swap under canonicalization:
    # Original black rights become "my" rights; original white rights become "opp" rights.
    myK = 1 if "k" in castling else 0
    myQ = 1 if "q" in castling else 0
    oppK = 1 if "K" in castling else 0
    oppQ = 1 if "Q" in castling else 0

    # En passant square rotates 180 degrees as well
    if ep == "-":
        ep_idx = -1
    else:
        ep2 = _rotate_180_square(ep)
        f, r = _square_to_xy(ep2)
        ep_idx = _board_index_a8_to_h1(f, r)

    return board_can, myK, myQ, oppK, oppQ, ep_idx


# =========================
# 2D Relative Attention (Shaw-style)
# =========================

class RelSelfAttention2DShaw(nn.Module):
    """
    Shaw-style 2D relative self-attention for fixed 8x8 board (T=64).
    e_ij = ((q_i + aQ_ij) · (k_j + aK_ij)) / sqrt(dh)
    z_i  = sum_j softmax(e_ij) * (v_j + aV_ij)

    Shapes:
      x: [B, T, d_model]
      returns: [B, T, d_model]
    """

    def __init__(self, d_model: int, n_heads: int, grid_hw=(8, 8), dropout_p: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        H, W = grid_hw
        assert (H, W) == (8, 8), "This module is currently tailored to 8x8 boards."

        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.dropout_p = dropout_p
        self.T = H * W

        dx_max = W - 1
        dy_max = H - 1
        self.num_rel = (2 * dx_max + 1) * (2 * dy_max + 1)

        self.Wq = nn.Linear(d_model, d_model, bias=bias)
        self.Wk = nn.Linear(d_model, d_model, bias=bias)
        self.Wv = nn.Linear(d_model, d_model, bias=bias)
        self.Wo = nn.Linear(d_model, d_model, bias=bias)

        # Per-head relative vectors
        self.rel_q = nn.Parameter(torch.empty(n_heads, self.num_rel, self.dh))
        self.rel_k = nn.Parameter(torch.empty(n_heads, self.num_rel, self.dh))
        self.rel_v = nn.Parameter(torch.empty(n_heads, self.num_rel, self.dh))

        # Precompute [T,T] relative-index table
        rel_index = self._build_rel_index(H, W, dx_max, dy_max)  # [64,64] long
        self.register_buffer("rel_index", rel_index, persistent=False)

        # Init
        nn.init.normal_(self.rel_q, std=0.02)
        nn.init.normal_(self.rel_k, std=0.02)
        nn.init.normal_(self.rel_v, std=0.02)

    @staticmethod
    def _build_rel_index(H: int, W: int, dx_max: int, dy_max: int) -> torch.Tensor:
        ys = torch.arange(H)
        xs = torch.arange(W)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # [T,2] as (x,y), y=0 is top row (rank 8)

        disp = coords[None, :, :] - coords[:, None, :]  # [T,T,2] (xj-xi, yj-yi)
        dx = disp[..., 0].clamp(-dx_max, dx_max) + dx_max
        dy = disp[..., 1].clamp(-dy_max, dy_max) + dy_max
        rel_id = dx + (2 * dx_max + 1) * dy
        return rel_id.long()

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, 64, d_model]
        attn_bias: optional additive bias broadcastable to [B, heads, 64, 64]
        """
        B, T, _ = x.shape
        if T != self.T:
            raise ValueError(f"Expected T={self.T}, got {T}")

        q = self.Wq(x).view(B, T, self.n_heads, self.dh).transpose(1, 2)  # [B,h,T,dh]
        k = self.Wk(x).view(B, T, self.n_heads, self.dh).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_heads, self.dh).transpose(1, 2)

        # Gather relative vectors: [h,T,T,dh]
        rel_q = self.rel_q[:, self.rel_index]  # [h,T,T,dh]
        rel_k = self.rel_k[:, self.rel_index]
        rel_v = self.rel_v[:, self.rel_index]

        # logits = (q + rel_q) · (k + rel_k)
        qk = torch.einsum("bhtd,bhsd->bhts", q, k)               # [B,h,T,T]
        q_relk = torch.einsum("bhtd,htsd->bhts", q, rel_k)       # [B,h,T,T]
        relq_k = torch.einsum("htsd,bhsd->bhts", rel_q, k)       # [B,h,T,T]
        relq_relk = torch.einsum("htsd,htsd->hts", rel_q, rel_k) # [h,T,T]

        logits = (qk + q_relk + relq_k + relq_relk.unsqueeze(0)) / math.sqrt(self.dh)

        if attn_bias is not None:
            logits = logits + attn_bias

        attn = torch.softmax(logits, dim=-1)
        if self.dropout_p and self.training:
            attn = F.dropout(attn, p=self.dropout_p)

        out_v = torch.einsum("bhts,bhsd->bhtd", attn, v)         # [B,h,T,dh]
        out_relv = torch.einsum("bhts,htsd->bhtd", attn, rel_v)  # [B,h,T,dh]
        out = out_v + out_relv

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)  # [B,T,d_model]
        return self.Wo(out)


class RelEncoderLayer(nn.Module):
    """
    A transformer encoder layer that uses 2D relative self-attention (Shaw-style) for the board tokens.
    Pre-norm by default (like your current norm_first=True usage).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 2048,
        dropout_p: float = 0.0,
        norm_first: bool = True,
        use_gelu: bool = False,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.attn = RelSelfAttention2DShaw(d_model=d_model, n_heads=n_heads, dropout_p=dropout_p)

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=True)

        act = nn.GELU() if use_gelu else nn.ReLU()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act,
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,64,d_model]
        if self.norm_first:
            y = self.attn(self.norm1(x))
            if self.dropout_p and self.training:
                y = F.dropout(y, p=self.dropout_p)
            x = x + y

            y = self.ff(self.norm2(x))
            if self.dropout_p and self.training:
                y = F.dropout(y, p=self.dropout_p)
            x = x + y
            return x

        # Post-norm variant (rarely needed)
        y = self.attn(x)
        if self.dropout_p and self.training:
            y = F.dropout(y, p=self.dropout_p)
        x = self.norm1(x + y)

        y = self.ff(x)
        if self.dropout_p and self.training:
            y = F.dropout(y, p=self.dropout_p)
        x = self.norm2(x + y)
        return x


# =========================
# New Chess Transformer (2D Relative)
# =========================

class ChessRelativeTransformer(nn.Module):
    """
    Drop-in replacement class name ("ChessRelativeTransformer") so you can swap imports,
    but it uses:
      - fixed tokens: [eval] + 64 board squares
      - 2D relative attention for the 64 squares
      - extra FEN info injected via additive embeddings to square embeddings
      - optional fusion transformer so eval token can aggregate board context

    emb_indices format (LongTensor):
      - shape: [B, 65]
      - emb_indices[:, 0] is eval token id (always 0)
      - emb_indices[:, 1:] are board piece ids in canonical stm view (0..12)
    """

    def __init__(
        self,
        bin_predictor: BinPredictor,
        d_model: int = 512,
        nhead: int = 8,
        board_num_layers: int = 8,
        fusion_num_layers: int = 2,
        dim_feedforward: int = 2048,
        norm_first: bool = True,
        use_gelu: bool = False,
        dropout_p: float = 0.0,
        use_square_abs_embedding: bool = True,
    ) -> None:
        super().__init__()

        self.bin_predictor = bin_predictor
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")

        # --- Token embedding ---
        # Board pieces: 13 tokens. We'll embed board squares from emb_indices[:, 1:].
        self.board_embedding = nn.Embedding(num_embeddings=BOARD_VOCAB_SIZE, embedding_dim=d_model)

        # Eval token embedding (learned)
        self.eval_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.eval_embedding, std=0.02)

        # Optional learned absolute square embedding (helps a lot in practice)
        self.use_square_abs_embedding = use_square_abs_embedding
        if use_square_abs_embedding:
            self.square_embedding = nn.Embedding(num_embeddings=64, embedding_dim=d_model)

        # --- Extra FEN state as additive embeddings (broadcast to squares) ---
        # Castling rights: we encode four bits (myK, myQ, oppK, oppQ) as embeddings and sum the "on" ones.
        self.castle_K_my = nn.Parameter(torch.empty(d_model))
        self.castle_Q_my = nn.Parameter(torch.empty(d_model))
        self.castle_K_opp = nn.Parameter(torch.empty(d_model))
        self.castle_Q_opp = nn.Parameter(torch.empty(d_model))
        for p in (self.castle_K_my, self.castle_Q_my, self.castle_K_opp, self.castle_Q_opp):
            nn.init.normal_(p, std=0.02)

        # En passant:
        # - global EP presence embedding (added to all squares; helps model "EP exists")
        # - per-square EP-marker embedding added ONLY at the EP square (helps model identify where)
        self.ep_global = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.ep_global, std=0.02)
        self.ep_square_marker = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.ep_square_marker, std=0.02)

        # --- Board encoder with 2D relative attention ---
        self.board_layers = nn.ModuleList([
            RelEncoderLayer(
                d_model=d_model,
                n_heads=nhead,
                dim_feedforward=dim_feedforward,
                dropout_p=dropout_p,
                norm_first=norm_first,
                use_gelu=use_gelu,
            )
            for _ in range(board_num_layers)
        ])

        # --- Fusion encoder (standard attention) ---
        # Lets the eval token gather from board tokens via normal attention.
        # This is cheap (seq len 65) and keeps your "eval token" pattern.
        if fusion_num_layers > 0:
            activation_type = "gelu" if use_gelu else "relu"
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=norm_first,
                activation=activation_type,
                dropout=dropout_p,
            )
            self.fusion_transformer = nn.TransformerEncoder(enc_layer, num_layers=fusion_num_layers)
        else:
            self.fusion_transformer = None

        # Final norm/head (same spirit as your current class)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj_to_bin_predictor_logits = nn.Linear(d_model, bin_predictor.total_num_bins)

        # Cache for soft CE
        self._soft_target_sigma_cache: Dict[Tuple[int, float], float] = {}

    # -------------------------
    # Public API (kept the same)
    # -------------------------

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
        # NOTE: Your original code calls CrossEntropyLoss with probs; that's not the usual signature.
        # I'm keeping the method as-is for drop-in compatibility.
        logits = self.compute_bin_predictor_logits(emb_indices)
        return self.cross_entropy_loss(logits / temperature, bin_class_probs)

    def compute_bin_predictor_logits(self, emb_indices: torch.LongTensor) -> torch.Tensor:
        """
        emb_indices: [B, 65]
          - [0] eval token (ignored except for shape; eval embedding is learned parameter)
          - [1:] board piece ids 0..12 in canonical stm view
        """
        if emb_indices.dim() != 2 or emb_indices.size(1) != 65:
            raise ValueError(f"Expected emb_indices shape [B,65], got {tuple(emb_indices.shape)}")

        B = emb_indices.size(0)

        # --- Build board embeddings ---
        board_ids = emb_indices[:, 1:]  # [B,64]
        board_x = self.board_embedding(board_ids)  # [B,64,d_model]

        # absolute square embedding (optional)
        if self.use_square_abs_embedding:
            square_ids = torch.arange(64, device=emb_indices.device).unsqueeze(0).expand(B, 64)
            board_x = board_x + self.square_embedding(square_ids)

        # --- Add broadcast "extra FEN info" that we encoded into the token IDs via reserved negative space ---
        # In this module, we pass those extras via internal buffers created by fen_list_to_emb_indices().
        # Specifically: we stash per-batch extras on the module during fen_list_to_emb_indices().
        extras = getattr(self, "_last_fen_extras", None)
        if extras is None or extras.device != emb_indices.device or extras.size(0) != B:
            raise RuntimeError(
                "Missing FEN extras. Use fen_list_to_emb_indices(fen_list, device) "
                "to generate emb_indices right before calling forward/logits."
            )

        # extras: [B, 5] => myK,myQ,oppK,oppQ,ep_idx
        myK, myQ, oppK, oppQ = extras[:, 0], extras[:, 1], extras[:, 2], extras[:, 3]
        ep_idx = extras[:, 4]  # -1 or 0..63

        # Castling broadcast
        castle_vec = torch.zeros(B, self.castle_K_my.numel(), device=emb_indices.device, dtype=board_x.dtype)
        castle_vec = castle_vec \
            + myK.unsqueeze(1).to(board_x.dtype) * self.castle_K_my.unsqueeze(0) \
            + myQ.unsqueeze(1).to(board_x.dtype) * self.castle_Q_my.unsqueeze(0) \
            + oppK.unsqueeze(1).to(board_x.dtype) * self.castle_K_opp.unsqueeze(0) \
            + oppQ.unsqueeze(1).to(board_x.dtype) * self.castle_Q_opp.unsqueeze(0)

        board_x = board_x + castle_vec.unsqueeze(1)

        # En passant: if exists, add global EP embedding to all squares and a marker to the EP square
        has_ep = (ep_idx >= 0)
        if has_ep.any():
            board_x = board_x + has_ep.to(board_x.dtype).unsqueeze(1).unsqueeze(2) * self.ep_global.view(1, 1, -1)
            # add marker to the ep square only
            # We'll scatter-add marker embedding to [B,64,d_model]
            marker = has_ep.to(board_x.dtype).unsqueeze(1) * self.ep_square_marker.view(1, -1)  # [B,d_model]
            idx = ep_idx.clamp(min=0)  # safe
            board_x[has_ep, idx[has_ep]] += marker[has_ep]

        # --- Board 2D relative encoder ---
        for layer in self.board_layers:
            board_x = layer(board_x)  # [B,64,d_model]

        # --- Build eval token + fuse ---
        eval_x = self.eval_embedding.expand(B, 1, -1)  # [B,1,d_model]
        seq = torch.cat([eval_x, board_x], dim=1)      # [B,65,d_model]

        if self.fusion_transformer is not None:
            seq = self.fusion_transformer(seq)

        # Pool eval token
        return self.proj_to_bin_predictor_logits(self.layer_norm(seq[:, 0, :]))

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
        """
        Produces emb_indices [B,65] and also stashes per-batch extras needed for embeddings:
          extras [B,5] = (myK,myQ,oppK,oppQ,ep_idx)
        """
        emb_batch: List[torch.LongTensor] = []
        extras_batch: List[torch.Tensor] = []

        for fen in fen_list:
            expanded, stm, castling, ep = _parse_fen_parts(fen)
            board_can, myK, myQ, oppK, oppQ, ep_idx = _canonicalize_to_stm_view(expanded, stm, castling, ep)

            # board_can is 64 chars in a8..h1 order already (because expansion follows FEN ranks 8->1)
            board_ids = torch.tensor([_PIECE_TO_INDEX[c] for c in board_can], dtype=torch.long)  # [64]

            # eval token id (dummy 0), then 64 board ids
            indices = torch.cat([torch.zeros(1, dtype=torch.long), board_ids], dim=0)  # [65]
            emb_batch.append(indices)

            extras_batch.append(torch.tensor([myK, myQ, oppK, oppQ, ep_idx], dtype=torch.long))

        emb = torch.stack(emb_batch, dim=0).to(device=device)  # [B,65]

        # Stash extras for the next forward/logits call (simple + fast for your current call pattern)
        self._last_fen_extras = torch.stack(extras_batch, dim=0).to(device=device)  # [B,5]
        return emb

    def compute_white_win_prob_from_fen(self, fen_list: Sequence[str], device: str) -> torch.Tensor:
        emb_indices = self.fen_list_to_emb_indices(fen_list, device)
        return self.compute_white_win_prob(emb_indices)

    def compute_white_win_prob(self, emb_indices: torch.LongTensor) -> torch.Tensor:
        # NOTE: This returns "avg bin index / denom" like your current class.
        # If your bins are centered and symmetric and you later want a true White POV probability,
        # you'll want to adjust this because we canonicalize to side-to-move POV.
        avg_bin_index = self.compute_avg_bin_index(emb_indices)
        denom = max(1, self.bin_predictor.total_num_bins - 1)
        return avg_bin_index / denom

    def _get_soft_target_sigma(self, num_classes: int, target_center_prob: float) -> float:
        cache_key = (num_classes, float(target_center_prob))
        if cache_key in self._soft_target_sigma_cache:
            return self._soft_target_sigma_cache[cache_key]

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
