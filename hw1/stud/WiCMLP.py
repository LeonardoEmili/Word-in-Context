from torch import nn
from typing import *
from stud.utils import *
import torch

class WiCMLP(nn.Module):
    ''' A simple MLP neural network for the WiC task. '''
    def __init__(
        self,
        vocabulary: int,
        embedding_size: int = 300,
        hidden_size: int = 256,
        vectors_store: torch.Tensor = None
    ) -> None:
        super().__init__()
        set_seed()  # Set the reproducibility seed
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        self.embedding = nn.Embedding.from_pretrained(vectors_store) if vectors_store else nn.Embedding(self.vocab_size, embedding_size, padding_idx=0)
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2 * self.embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        X1_lengths: torch.Tensor,
        X2_lengths: torch.Tensor
    ) -> torch.Tensor:
        # Retrieve token word embeddings
        X1 = self.dropout(self.embedding(X1))
        X2 = self.dropout(self.embedding(X2))

        # Aggregate sentence word embeddings ignoring the padding elements
        X1 = torch.stack([x[:length].mean(dim=0) for x,length in zip(X1, X1_lengths)])
        X2 = torch.stack([x[:length].mean(dim=0) for x,length in zip(X2, X2_lengths)])

        # Concatenate the obtained representations
        X = torch.cat((X1, X2), dim=-1)

        # Pass it through two fully connected layers
        X = torch.relu(self.dropout(self.fc1(X)))
        X = torch.sigmoid(self.dropout(self.fc2(X)).squeeze(1))
        return X