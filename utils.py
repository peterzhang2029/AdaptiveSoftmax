import urllib
import shutil
import zipfile
import os
import pickle
from collections import Counter
import numpy as np
import codecs


def build_dict(train_file_path):
    vocab = Counter()

    print('Constructing dictionary...')

    with open(train_file_path,'r') as f:
        for line in f:
            lines = line.strip().split()
            for word in lines:
                word = word.strip()
                if word:
                    vocab[word] += 1
    vocab['</s>'] = 1
    vocab_sorted = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    wordmap = {k: id for id, (k, _) in enumerate(vocab_sorted)}
    return wordmap


def get_batch(file_path, word_dict, batch_size=128, bptt = 20):
    UNK_IDX = word_dict['<UNK>']

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            line = [word_dict.get(w, UNK_IDX) for w in line]
            data.extend(line)
    data = np.array(data)

    data_cut = data[:data.shape[0] // (batch_size * bptt) * bptt * batch_size]

    data_next = data_cut.copy()
    data_next[:-1] = data_cut[1:]
    data_next[-1] = word_dict['</s>']
    
    input = np.array_split(data_cut.reshape((batch_size, -1)),
                           data_cut.shape[0] / batch_size / bptt, axis=1)
    label = np.array_split(data_next.reshape((batch_size, -1)),
                           data_next.shape[0] / batch_size / bptt, axis=1)

    return input, label



