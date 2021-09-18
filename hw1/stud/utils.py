import torch
import pickle
from torch import nn
from tqdm import tqdm
from typing import *
import numpy as np
import random as rnd
from collections import namedtuple, defaultdict
from torch.nn.utils.rnn import pad_sequence
from nltk.stem import WordNetLemmatizer

import nltk
from nltk import word_tokenize

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rnd.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_vocab(path: str = 'model/vocab_lemma.pkl', unk_index: int = 1) -> defaultdict:
    with open(path, 'rb') as f:
         vocab = defaultdict(lambda: unk_index, pickle.load(f))
    return vocab

# Useful namedtuple to hold token's information
Token = namedtuple('Token', 'text lemma pos')

def parse_sentence(row_sent: List[Dict]) -> Tuple[List[Token]]:
    '''
    Returns the sentence pair (s1, s2) that contains a list of parsed tokens.
    '''
    _, lemma, pos, s1, s2, start1, end1, start2, end2 = row_sent.values()
    start1, end1, start2, end2 = map(int, [start1, end1, start2, end2])
    target_word_1, target_word_2 = s1[start1:end1], s2[start2:end2]

    def _parse_sentence(s: str) -> List[Token]:
        ''' Returns the collection of tokens for a given sentence [s]. '''
        lemmatizer = WordNetLemmatizer()
        return [Token(t, lemmatizer.lemmatize(t), None) for t in word_tokenize(s)]

    s1_prefix, s1_suffix = _parse_sentence(s1[:start1]), _parse_sentence(s1[end1:])
    parsed_s1 = s1_prefix \
                + [Token(target_word_1, lemma, pos)] \
                + s1_suffix

    s2_prefix, s2_suffix = _parse_sentence(s2[:start2]), _parse_sentence(s2[end2:])
    parsed_s2 = s2_prefix \
                + [Token(target_word_2, lemma, pos)] \
                + s2_suffix
    return parsed_s1, parsed_s2

def parse_sentences(
    sentence_pairs: List[Dict],
    tag: str = 'test',
    vectorize_fn = None,
    word2idx: Dict[str, int] = None
    ) -> List[Tuple[List[Token]]]:
    '''
    Returns the collection of sentence pairs, where for each token in the sentence we have the named tuple
    (text, lemma, pos), where pos is None iff that is the target token.
    Please note that START_SEP and END_SEP can be any pair of token that is not present in the input 
    '''
    parsed_sentences = []
    _word2idx = word2idx.copy()
    for pair in sentence_pairs:
        s1, s2 = parse_sentence(pair)
        if vectorize_fn is not None:
            s1, s2 = vectorize_fn(s1, s2, _word2idx)
        parsed_sentences.append((s1, s2))
    return parsed_sentences

def vectorize_sentence_pair(
    s1: List[Token],
    s2: List[Token],
    word2idx: Dict[str, int]
    ) -> Tuple[List[Token], List[Token]]:
    return [word2idx[t.lemma] for t in s1], [word2idx[t.lemma] for t in s2]

def simple_collate_fn(
    batch: List[Tuple[List[Token],
    List[Token], bool]],
    padding_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:    

    sentence_pair_lengths = [(len(s1),len(s2)) for s1,s2,*_ in batch]
    X1_lengths, X2_lengths = map(torch.tensor, zip(*sentence_pair_lengths))

    X1, X2 = zip(*[(torch.tensor(s1),torch.tensor(s2)) for s1,s2 in batch])
    X1 = pad_sequence(X1, batch_first=True, padding_value=padding_idx)
    X2 = pad_sequence(X2, batch_first=True, padding_value=padding_idx)

    return (X1, X2), (X1_lengths, X2_lengths)

def move_to_device(device: str, *args: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
    ''' Moves a collections of tensors to the specified device if they are not already there. '''
    return tuple(arg.to(device) if str(arg.device) != device else arg for arg in args)
