import nltk

from config import args
from utils_c import *
import sklearn
import numpy as np
import tensorflow as tf


class BaseDataLoader(object):
    def __init__(self):
        self.enc_inp = None
        self.dec_inp = None # word dropout
        self.dec_out = None
        self.labels = None
        self.params = {'vocab_size': None, 'word2idx': None, 'idx2word': None,
                       '<start>': None, '<end>': None}

    def next_batch(self):
        for i in range(0, len(self.enc_inp), args.batch_size):
            yield (self.enc_inp[i : i + args.batch_size],
                   self.dec_inp[i : i + args.batch_size],
                   self.dec_out[i : i + args.batch_size],
                   self.labels[i : i + args.batch_size],)

    def get_word_idx(self, word):
        if word in self.params['word2idx'] and self.params['word2idx'][word] < args.vocab_size:
            return self.params['word2idx'][word]
        else:
            return self.params['word2idx']['<unk>']


class DataLoader(BaseDataLoader):
    def __init__(self, skip=None, file_path=None, num_split=1, two_gram=False):
        if file_path is None:
            file_path = args.data_path
        self.file_path = file_path
        self.num_split = num_split
        self.two_gram = two_gram

        if skip is None:
            super().__init__()
            self._index_from = 0
            if file_path is None: file_path = args.data_path
            self.file_path = file_path
            self.params['word2idx'] = self._load_word2idx()
            self.params['idx2word'] = self._load_idx2word()
            self.params['vocab_size'] = args.vocab_size
            self.params['<start>'] = self.params['word2idx']['<start>']
            self.params['<end>'] = self.params['word2idx']['<end>']

            self.enc_inp, self.dec_inp_full, self.dec_out, self.labels, self.scores = self._load_data()
            self.dec_inp = self._word_dropout(self.dec_inp_full)


    def _load_data(self):
        
        self.X, self.y, self.scores = self.read_file()
        self.range_ = range(len(self.X))

        X = self.get_train_x(self.X)
        self.train_x = X
        self.y = self.get_train_y(self.y)
        self.scores = self.get_train_y(self.scores)
        y = self.y
        tran_len_ = int(len(X) * 0.9)

        X_train, y_train, X_test, y_test, s_train, s_test = X[:tran_len_], y[:tran_len_], X[tran_len_:], y[tran_len_:], self.scores[:tran_len_], self.scores[tran_len_:]
        print("Data Loaded")
        X_tr_enc_inp, X_tr_dec_inp, X_tr_dec_out, y_tr, s_tr = self._pad(X_train, y_train, s_train)
        X_te_enc_inp, X_te_dec_inp, X_te_dec_out, y_te, s_te = self._pad(X_test, y_test, s_test)
        enc_inp = np.concatenate((X_tr_enc_inp, X_te_enc_inp))
        dec_inp = np.concatenate((X_tr_dec_inp, X_te_dec_inp))
        dec_out = np.concatenate((X_tr_dec_out, X_te_dec_out))
        labels = np.concatenate((y_tr, y_te))
        scores = np.concatenate((s_tr, s_te))
        print("Data Padded")
        return enc_inp, dec_inp, dec_out, labels, scores

    def get_train_x(self, X):
        result = []
        for x in X:
            temp = []
            words = nltk.tokenize.word_tokenize(x)
            for word in words:
                if word in self.params['word2idx'] and self.params['word2idx'][word] < args.vocab_size:
                    temp.append(self.params['word2idx'][word])
                elif word in self.params['word2idx']:
                    temp.append(self.params['word2idx']['<unk>'])
            result.append(temp)
        return np.array(result)

    def get_train_y(self, y):
        auth_class = list(set(y))
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(auth_class)
        y_numeric = le.transform(y)
        return np.array(y_numeric)

    def read_file(self):
        author = []
        content = []
        score = []
        with open(self.file_path, 'r', encoding="utf8", errors='ignore') as f:
            for line in f:
                # print(line)
                per_line = line.split(" ", self.num_split)
                if len(line) < 5:
                    continue
                
                author.append(int(per_line[0]))
                if len(per_line) > 2:
                    score.append(int(per_line[1]))
                else:
                    score.append(int(1))
                if len(per_line) > 2:
                    content.append(per_line[2])
                else:
                    content.append(per_line[1])

        return np.array(content), np.array(author), np.array(score, dtype=np.int32)

    def _pad(self, X, y, score):
        _pad = self.params['word2idx']['<pad>']
        _start = self.params['word2idx']['<start>']
        _end = self.params['word2idx']['<end>']

        enc_inp = []
        dec_inp = []
        dec_out = []
        labels = []
        scores = []
        
        for i, x in enumerate(X):
            _y = y[i]
            _score = score[i]
            x = x[1:]
            if len(x) < args.max_len:
                enc_inp.append(x + [_pad] * (args.max_len-len(x)))
                dec_inp.append([_start] + x + [_pad] * (args.max_len-len(x)))
                dec_out.append(x + [_end] + [_pad] * (args.max_len-len(x)))
                labels.append(_y)
                scores.append(_score)
            else:
                truncated = x[:args.max_len]
                enc_inp.append(truncated)
                dec_inp.append([_start] + truncated)
                dec_out.append(truncated + [_end])
                labels.append(_y)
                scores.append(_score)

                if len(x) > 1.8 * args.max_len:

                    truncated = x[-args.max_len:]
                    enc_inp.append(truncated)
                    dec_inp.append([_start] + truncated)
                    dec_out.append(truncated + [_end])
                    labels.append(_y)
                    scores.append(_score)

        return np.array(enc_inp), np.array(dec_inp), np.array(dec_out), np.array(labels), np.array(scores)

    def _load_word2idx(self):
        if args.use_my_dict:
            word2idx = get_my_word2id()
        else:
            word2idx = tf.contrib.keras.datasets.imdb.get_word_index()
        print("Word Index Loaded")
        word2idx = {k: (v+self._index_from) for k, v in word2idx.items()}
        word2idx['<pad>'] = 0
        word2idx['<start>'] = 1
        word2idx['<unk>'] = 2
        word2idx['<end>'] = 3
        return word2idx

    def _load_idx2word(self):
        idx2word = {i: w for w, i in self.params['word2idx'].items()}
        return idx2word

    def _word_dropout(self, x):
        is_dropped = np.random.binomial(1, args.word_dropout_rate, x.shape)
        fn = np.vectorize(lambda x, k: self.params['word2idx']['<unk>'] if (k and (x not in range(4))) else x)
        return fn(x, is_dropped)

    def shuffle(self):
        self.enc_inp, self.dec_inp, self.dec_out, self.dec_inp_full, self.labels, self.scores = sklearn.utils.shuffle(
            self.enc_inp, self.dec_inp, self.dec_out, self.dec_inp_full, self.labels, self.scores)

    def update_word_dropout(self):
        self.dec_inp = self._word_dropout(self.dec_inp_full)

    def sample_test(self):
        index = np.random.permutation(self.range_)[:args.test_sample_size]
        return self.X[index], self.y[index], self.scores[index]

    def next_batch(self):
        for i in range(0, len(self.enc_inp), args.batch_size):
            yield (self.enc_inp[i : i + args.batch_size],
                   self.dec_inp[i : i + args.batch_size],
                   self.dec_out[i : i + args.batch_size],
                   self.labels[i : i + args.batch_size],
                   self.scores[i : i + args.batch_size],)
