import json
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from dataclasses import dataclass, field
from typing import List
import mmap
from src.bin_predictor import BinPredictor
from src.dataset.fen_utils import expand_fen_string


FEN_CHAR_TO_INDEX = {
    'eval': 0,
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
    '1': 13, '2': 14, '3': 15, '4': 16, '5': 17, '6': 18, '7': 19, '8': 20, '9': 21,
    '/': 22, '.': 23, ' ': 24, 'w': 25, 'W': 26,
    '-': 27, 'd': 28, 'c': 29, 'f': 30, 'g': 31, 'h': 32, 'a': 33, 'e': 34, 'F': 35, 'E': 36
}
MAX_FEN_LENGTH = 90


@dataclass
class RawIO:
    expanded_fen: str
    cp_eval: int
    contains_eval: bool
    mate: int  # number of moves till mate, if negative then black mates white. If 0 no mate was found


class JSONLinesChessDataset(Dataset):
    """The lichess evaluation dataset."""
    def __init__(self, file_path, index_file='index.npy'):
        self.file_path = file_path
        self.index_file = index_file
        self.offsets = self._load_or_build_index()
        self._map_file()

    def _map_file(self):
        # Open the file and memory-map it
        with open(self.file_path, 'r+b') as f:  # Open the file in read-binary mode for mmap
            self.data = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)

    def _load_or_build_index(self):
        if os.path.exists(self.index_file):
            print('Loading index from file.')
            return np.memmap(self.index_file, mode='r', dtype=np.int64)
        else:
            print('Creating dataset index file.')
            offsets = [0]
            with open(self.file_path, 'rb') as file:
                pbar = tqdm(desc="Indexing", unit="lines")
                while line := file.readline():
                    offsets.append(file.tell())
                    pbar.update(1)
                pbar.close()

            offsets_array = np.asarray(offsets, dtype=np.int64)
            memmap = np.memmap(self.index_file, mode='w+', dtype=np.int64, shape=offsets_array.shape)
            memmap[:] = offsets_array[:]
            memmap.flush()
            print(f'Index file saved to: {self.index_file}')
            return np.memmap(self.index_file, mode='r', dtype=np.int64)

    def __len__(self):
        # The total number of items is the number of offsets minus one (for the end of the file)
        return len(self.offsets) - 1

    def __getitem__(self, idx):
        # Ensure the index is within bounds
        if idx >= len(self) or idx < 0:
            raise IndexError('Index out of bounds')

        # Calculate the start and end positions for the requested item
        start = self.offsets[idx]
        end = self.offsets[idx + 1]

        # Directly slice the mmap object to get the bytes for the desired line
        line_bytes = self.data[start:end]

        # Decode the bytes to a string, strip newline characters, and parse JSON
        line = line_bytes.decode('utf-8').strip()
        json_object = json.loads(line)

        fen_str = json_object['fen']
        expanded_fen_str = expand_fen_string(fen_str)
        pv = json_object['evals'][0]['pvs'][0]
        if 'cp' in pv:
            centipawn_eval = pv['cp']
            contains_eval = True
        else:
            centipawn_eval = 100 * 100 if pv['mate'] > 0 else -100 * 100
            contains_eval = False

        return RawIO(
            expanded_fen=expanded_fen_str,
            cp_eval=centipawn_eval,
            contains_eval=contains_eval,
            mate=pv.get('mate', 0),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the mmap object from the state
        del state['data']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the mmap object
        self._map_file()


class JSONLinesLichessDataset(Dataset):
    """Dataset of lichess evaluations extracted from lichess games databases."""

    def __init__(self, file_path, index_file=None):
        self.file_path = file_path
        if index_file is None:
            base_name, _ = os.path.splitext(file_path)
            self.index_file = f'{base_name}_index.npy'
        else:
            self.index_file = index_file
        self.offsets = self._load_or_build_index()
        self._map_file()

    def _map_file(self):
        # Open the file and memory-map it
        with open(self.file_path, 'r+b') as f:  # Open the file in read-binary mode for mmap
            self.data = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)

    def _load_or_build_index(self):
        if os.path.exists(self.index_file):
            print('Loading index from file.')
            return np.memmap(self.index_file, mode='r', dtype=np.int64)
        else:
            print('Creating dataset index file.')
            offsets = [0]
            with open(self.file_path, 'rb') as file:
                pbar = tqdm(desc="Indexing", unit="lines")
                while line := file.readline():
                    offsets.append(file.tell())
                    pbar.update(1)
                pbar.close()

            offsets_array = np.asarray(offsets, dtype=np.int64)
            memmap = np.memmap(self.index_file, mode='w+', dtype=np.int64, shape=offsets_array.shape)
            memmap[:] = offsets_array[:]
            memmap.flush()
            print(f'Index file saved to: {self.index_file}')
            return np.memmap(self.index_file, mode='r', dtype=np.int64)

    def __len__(self):
        # The total number of items is the number of offsets minus one (for the end of the file)
        return len(self.offsets) - 1

    def __getitem__(self, idx):
        # Ensure the index is within bounds
        if idx >= len(self) or idx < 0:
            raise IndexError('Index out of bounds')

        # Calculate the start and end positions for the requested item
        start = self.offsets[idx]
        end = self.offsets[idx + 1]

        # Directly slice the mmap object to get the bytes for the desired line
        line_bytes = self.data[start:end]

        # Decode the bytes to a string, strip newline characters, and parse JSON
        line = line_bytes.decode('utf-8').strip()
        json_object = json.loads(line)

        fen_str = json_object['FEN']
        expanded_fen_str = expand_fen_string(fen_str)

        if 'cp' in json_object:
            centipawn_eval = json_object['cp']
            contains_eval = True
        else:
            centipawn_eval = 100 * 100 if json_object['mate'] > 0 else -100 * 100
            contains_eval = False

        return RawIO(
            expanded_fen=expanded_fen_str,
            cp_eval=centipawn_eval,
            contains_eval=contains_eval,
            mate=json_object.get('mate', 0),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the mmap object from the state
        del state['data']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the mmap object
        self._map_file()


@dataclass
class TransformerIO:
    embedding_indices: torch.IntTensor  # (B, Length of FEN + 1). We pad FEN strings to 85 characters
    cp_evals: torch.LongTensor  # (B)
    cp_valid: torch.BoolTensor  # (B), false if cp_eval was missing (because mate was found)
    expanded_fens: List[str]
    bin_classes: torch.LongTensor  # Bin predictor class of (cp_evals, mate)
    white_win_prob: torch.Tensor = field(init=False)  # (B)

    def __post_init__(self):
        # For conversion of centipawns to win percentage see https://lichess.org/page/accuracy
        # Win% = 50 + 50 * (2 / (1 + exp(-0.00368208 * centipawns)) - 1)
        self.white_win_prob = 1.0 / (1.0 + torch.exp(-0.00368208 * self.cp_evals))


    def to(self, device):
        return TransformerIO(
            embedding_indices=self.embedding_indices.to(device=device),
            cp_evals=self.cp_evals.to(device=device),
            cp_valid=self.cp_valid.to(device=device),
            expanded_fens=self.expanded_fens,
            bin_classes=self.bin_classes.to(device=device)
        )


def collate_fn(batch: List[RawIO], bin_predictor: BinPredictor) -> TransformerIO:
    """Batch is a list of dicts with key"""
    emb_indices_batch = []
    bin_classes = []
    # Use a single token for the eval
    for batch_elt in batch:
        # print(batch_elt.expanded_fen)
        try:
            indices_lst = [FEN_CHAR_TO_INDEX['eval']] + [FEN_CHAR_TO_INDEX[c] for c in batch_elt.expanded_fen]
        except KeyError:
            print(f'Failed to tokenize: {batch_elt.expanded_fen}')
            exit(1)
        # Pad to 85 characters
        indices_lst += [FEN_CHAR_TO_INDEX[' '] for _ in range(MAX_FEN_LENGTH - len(batch_elt.expanded_fen))]
        emb_indices_batch.append(
            torch.tensor(indices_lst, dtype=torch.long)
        )
        bin_classes.append(bin_predictor.to_bin_index(batch_elt.cp_eval, batch_elt.mate))

    eval_batch = torch.tensor([x.cp_eval for x in batch], dtype=torch.long)
    cp_valid = torch.tensor([x.contains_eval for x in batch], dtype=torch.bool)
    embedding_indices = torch.stack(emb_indices_batch, dim=0)
    return TransformerIO(
        embedding_indices=embedding_indices,
        cp_evals=eval_batch,
        cp_valid=cp_valid,
        expanded_fens=[x.expanded_fen for x in batch],
        bin_classes=torch.tensor(bin_classes, dtype=torch.long)
    )
