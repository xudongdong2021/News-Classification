from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from collections import Counter
from nltk import word_tokenize


# tokenize and build vocab
def load_tokenize_and_build_vocab(args):
    # read data
    df = pd.read_csv(args.data_path, usecols=['text', 'label'])
    df['text'] = df['text'].astype(str)
    # tokenize
    tokens = []
    for text in df['text']:
        tokens.extend(word_tokenize(text))
    # build vocab
    word_counter = Counter(tokens)
    vocab = sorted(word_counter, key=word_counter.get, reverse=True)[:args.vocab_size-2]
    vocab.append('<UNK>')
    vocab.append('<PAD>')

    word_index = {word: i for i, word in enumerate(vocab)}
    return df, word_index

# create dataset
class NewsDataset(Dataset):
    def __init__(self, args, texts, word_index, labels=None):
        self.texts = texts
        self.labels = labels
        self.word_index = word_index
        self.seq_len = args.sequence_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        tokens = word_tokenize(text)
        sequence = [self.word_index.get(token, self.word_index['<UNK>']) for token in tokens]
        # truncate if length > seq_len else add PAD
        sequence = sequence[:self.seq_len] if len(sequence) > self.seq_len else sequence + [self.word_index['<PAD>']] * (self.seq_len - len(sequence))
        if self.labels is not None:
            return np.array(sequence), self.labels[idx]
        else:
            return np.array(sequence)



