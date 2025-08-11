import numpy as np
import torch
from MyNLPDataSet import MyNLPDataSet
from WikitextDataset import WikitextDataset
from torch.utils.data import DataLoader
import gzip
from data_utils import Corpus
from data_utils import get_lm_corpus
import argparse

#following commented functions are for character level modeling----------
def decode_token_char(token): # convert token to character
    return str(chr(max(32, token)))

def decode_tokens_char(tokens): # convert sequence of characters to tokens
    return ''.join(list(map(decode_token_char, tokens)))
#------------------------------------------------------------------------

def cycle(loader):
    while True:
        for data in loader:
            yield data

def get_loaders_enwiki8_basedon_transformerXL(seq_len, batch_size):
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
    parser.add_argument('--data', type=str, default='./enwik8',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='enwik8',
                        choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                        help='dataset name')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train','all', 'valid', 'test'],
                        help='which split to evaluate')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--tgt_len', type=int, default=5,
                        help='number of tokens to predict')


    args = parser.parse_args()

    corpus = get_lm_corpus(args.data, args.dataset)
    ntokens = len(corpus.vocab)
    print('length of dictionary =',ntokens)

    train_dataset = MyNLPDataSet(corpus.train, seq_len)
    val_dataset   = MyNLPDataSet(corpus.valid, seq_len)
    test_dataset   = MyNLPDataSet(corpus.test, seq_len)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = batch_size))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = batch_size))
    test_loader    = cycle(DataLoader(test_dataset, batch_size = batch_size))
    return train_loader, val_loader, test_loader, val_dataset

def get_loaders_enwiki8(seq_len, batch_size):
    # ---------prepare enwik8 data-----------
    with gzip.open('./data/enwik8.gz') as file:
        data = np.fromstring(file.read(int(95e6)), dtype = np.uint8)
        data_train, data_val = map(torch.from_numpy, np.split(data, [int(90e6)]))

    train_dataset = MyNLPDataSet(data_train, seq_len)
    val_dataset   = MyNLPDataSet(data_val, seq_len)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = batch_size))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = batch_size))
    test_loader    = cycle(DataLoader(val_dataset, batch_size = batch_size))
    return train_loader, val_loader, test_loader, val_dataset

def get_loaders_text8(seq_len, batch_size):
    # ---------prepare enwik8 data-----------
    #file_train = open('./data/text8/train.txt', 'r')
    file_train = open('./data/text8files/text8.train.txt', 'r')
    data_train = torch.from_numpy(np.fromstring(file_train.read(),dtype=np.uint8))
    #file_val = open('./data/text8/valid.txt', 'r')
    file_val = open('./data/text8files/text8.dev.txt', 'r')
    data_val = torch.from_numpy(np.fromstring(file_val.read(),dtype=np.uint8))
    #file_test = open('./data/text8/test.txt', 'r')
    file_test = open('./data/text8files/text8.test.txt', 'r')
    data_test = torch.from_numpy(np.fromstring(file_test.read(),dtype=np.uint8))

    train_dataset = MyNLPDataSet(data_train, seq_len)
    val_dataset   = MyNLPDataSet(data_val, seq_len)
    test_dataset   = MyNLPDataSet(data_test, seq_len)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = batch_size))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = batch_size))
    test_loader    = cycle(DataLoader(test_dataset, batch_size = batch_size))
    return train_loader, val_loader, test_loader, val_dataset

def get_loaders_wikitext_103(tokenizer, seq_len, batch_size):
    file_path_train = "data/wikitext-103/wiki.train.tokens"
    file_path_valid = "data/wikitext-103/wiki.valid.tokens"
    file_path_test = "data/wikitext-103/wiki.test.tokens"
    train_dataset = WikitextDataset(tokenizer, file_path_train,"TRAIN",seq_len=seq_len)
    val_dataset = WikitextDataset(tokenizer, file_path_valid,"VALID",seq_len=seq_len)
    test_dataset = WikitextDataset(tokenizer, file_path_test,"TEST",seq_len=seq_len)
    train_loader = cycle(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
    val_loader = cycle(DataLoader(val_dataset, batch_size=batch_size, shuffle=False))
    test_loader = cycle(DataLoader(test_dataset, batch_size=batch_size, shuffle=False))
    return train_loader, val_loader, test_loader, val_dataset, test_dataset