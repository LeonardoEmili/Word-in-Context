# Word-in-Context disambiguation

[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeonardoEmili/Word-in-Context/blob/main/hw1/stud/notebook.ipynb)

Word-in-Context (WiC) disambiguation as a binary classification task using static word embeddings (i.e. Word2Vec and GloVe) to determine whether words in different contexts have the same meaning.

## Implementation details
We propose a Bi-LSTM architecture with pre-trained word embeddings and test it against a simpler feed-forward neural network. For further insights, read the dedicated [report](https://github.com/LeonardoEmili/Word-in-Context/blob/main/report.pdf) or the [presentation slides](https://github.com/LeonardoEmili/Word-in-Context/blob/main/slides.pdf) (pages 2-6).

## Get the dataset
You may download the original dataset from [here](https://github.com/SapienzaNLP/nlp2021-hw1/tree/main/data).

## Test the model
For ready-to-go usage, simply run the notebook on Colab. In case you would like to test it on your local machine, please follow the [installation guide](https://github.com/SapienzaNLP/nlp2021-hw1#requirements). You may find the pre-trained models in `model/pretrained`.
