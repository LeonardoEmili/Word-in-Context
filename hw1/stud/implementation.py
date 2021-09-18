import torch
import numpy as np
from typing import List, Tuple, Dict
from stud.WiCLSTM import WiCLSTM
from stud.WiCMLP import WiCMLP
from stud.WiCDataset import WiCDataset
from torch.utils.data import DataLoader
from stud.utils import *

from model import Model

import nltk

def build_model(device: str) -> Model:
    # Flag to control which model to test, set it to false for the MLP
    use_lstm = True
    vocabulary_path = 'model/pretrained/vocab_lstm.pkl' if use_lstm else 'model/pretrained/vocab_mlp.pkl'
    weights_path = 'model/pretrained/lstm.pt' if use_lstm else 'model/pretrained/mlp.pt'

    # Define the collection of hyperparameters
    hparams = {'embedding_size': 300, 'hidden_size': 256}

    # Load the vocabulary
    vocabulary = load_vocab(vocabulary_path)

    # Load model weights and set it to evaluation mode
    model = WiCLSTM(vocabulary, **hparams) if use_lstm else WiCMLP(vocabulary, **hparams)
    model = model.to(device)
    model_params = torch.load(weights_path, map_location=device)
    model.load_state_dict(model_params['model_state_dict'])
    model.eval()

    # Download required data to perform tokenization and lemmatization
    nltk.download('punkt')
    nltk.download('wordnet')
    return StudentModel(device, model)


class RandomBaseline(Model):
    options = [
        ('True', 40000),
        ('False', 40000),
    ]

    def __init__(self):
        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        return [str(np.random.choice(self._options, 1, p=self._weights)[0]) for x in sentence_pairs]


class StudentModel(Model):
    '''
    Simple class that takes as input the device where to load
    tensor data and the model to use for the predict method.
    '''

    def __init__(self, device, model):
        self.device = device
        self.model = model

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        # Create a dataset and (simple) dataloader objects to read input sentences
        ds = WiCDataset(sentence_pairs, vectorize_sentence_pair, self.model.vocabulary)
        test_dataloader = DataLoader(ds, batch_size=1, collate_fn=simple_collate_fn)
        predictions = []
        for (x1, x2), (x1_lengths, x2_lengths) in test_dataloader:
            # We don't need to calculate the gradients of the loss function
            with torch.no_grad():
                x1, x2 = move_to_device(self.device, x1, x2)
                pred = self.model(x1, x2, x1_lengths, x2_lengths)
                predictions.append(str(pred.round().item() == 1))
        
        return predictions
