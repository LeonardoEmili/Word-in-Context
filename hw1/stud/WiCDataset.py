import torch
from typing import *
from torch.utils.data import Dataset, DataLoader
from stud.utils import parse_sentences

class WiCDataset(Dataset):
    def __init__(
        self,
        df: List[Dict],
        vectorize_fn, 
        word2idx: Dict[str, int],
        tag: str = 'test'
        ) -> None:
        self.sents = parse_sentences(df, tag, vectorize_fn=vectorize_fn, word2idx=word2idx)

    def __len__(self) -> int:
        return len(self.sents)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sents[idx]
