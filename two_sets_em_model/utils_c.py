import os
import string
import urllib
import io
import tensorflow as tf
import datetime
import logging
import numpy as np
import zipfile

from config import *


T_EOS = "<end>"
T_GO = "<start>"
T_UNK = "<unk>"
T_PAD = "<pad>"

DEF_DICT = {T_PAD: 0, T_UNK: 2, T_EOS: 3, T_GO: 1}
DEF_DICT_REV = {DEF_DICT[k]: k for k in DEF_DICT}
DEF_LS = [T_PAD, T_GO, T_UNK, T_EOS]


def inverse_sigmoid(x):
    return 1 / (1 + tf.exp(x))

def get_time_str():
    now = datetime.datetime.now()
    time_str = now.isoformat().split('.')[0]
    time_str = time_str.replace('-', '_')
    time_str = time_str.replace(':', '_')
    return time_str


def get_info_log(file_name, name='disc'):
    if not os.path.exists(file_name):
        with open(file_name, 'w'): pass
    parser_logger = logging.getLogger(name)
    parser_logger.addHandler(
        logging.FileHandler(file_name, 'a', 'utf-8', delay=False))
    parser_logger.setLevel(logging.INFO)
    return parser_logger


def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according
    to the definition of the dot product
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    a[a == np.inf] = 0
    b[b == np.inf] = 0
    
    norm_a = a / np.linalg.norm(a)
    norm_b = b / np.linalg.norm(b)
    res = np.dot(norm_a, norm_b)
    if np.isnan(res):
        return 0
    return res


def check_and_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_and_make_file_path(file_path):
    path = os.path.split(file_path)[0]
    check_and_make_dir(path)


def get_my_word2id():
    print('use my word2id')
    import json
    word2id = {}
    with open(args.dict_path) as handle:
        pairs = json.loads(handle.read())
        for i, pair in enumerate(pairs):
            word2id[pair] = i
    return word2id


def getfile(theurl, thedir):
    of = os.path.split(theurl)[-1]
    of = os.path.join(thedir, of)
    if os.path.exists(of):
        fname = of
    else:
        print("Downloading...", theurl)
        fname, _ = urllib.request.urlretrieve(theurl, of)
        print("... Downloading done")
    return fname


def get_variable(name, shape, init_zeros=False,
        regularizer=None, initializer=None):
    if init_zeros:
        var_init = tf.zeros_initializer()
    elif initializer is None:
        var_init = tf.contrib.layers.xavier_initializer()
    else:
        var_init = initializer

    return tf.get_variable(name, shape=shape,
                               initializer=var_init,
                               regularizer=regularizer)


def load_embedding_layer(trainable=True, id2w=None):

    vocab = []
    embedding_dim = args.embedding_dim
    embd_new = []
    for i in range(4, args.vocab_size):
        vocab.append(id2w[i])
        embd_new.append(np.random.random(embedding_dim))

    vocab_size = len(vocab)

    embedding = np.asarray(embd_new)

    w = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=trainable, name="W-GloVe-Embd")
    embedding_placeholder = tf.placeholder(
        tf.float32, [vocab_size, embedding_dim])
    embedding_init = w.assign(embedding_placeholder)

    def initializer(sess):
        sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

    vocab = DEF_LS + vocab
    wu = get_variable('special_tokens',
                      shape=[len(DEF_LS), embedding_dim])
    wu = tf.concat([wu, w], axis=0)

    word2id = {vocab[ind]: ind for ind in range(0, len(vocab))}
    id2word = {i: vocab[i] for i in range(len(vocab))}

    return initializer, wu, word2id, id2word, embd_new


def isAn(token):
    low = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    upper = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return token in low or token in upper or token in num


def load_partial_glove_layer(trainable=False, id2w=None):
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    of = getfile(url, "<path>")
    vocab = []
    embd = []
    with zipfile.ZipFile(of) as z:
        with io.TextIOWrapper(io.BytesIO(z.read('glove.6B.100d.txt')),
                              encoding='utf-8') as of:
            for line in of:
                row = line.strip().split(' ')
                vocab.append(row[0])
                embd.append(row[1:])

    temp_word2id = {vocab[ind]: ind for ind in range(0, len(vocab))}
    embedding_dim = len(embd[0])
    if id2w is None:
        id2w = vocab

    vocab_new = []
    embd_new = []
    for i in range(4, args.vocab_size):
        vocab_new.append(id2w[i])
        if id2w[i] not in temp_word2id:
            embd_new.append(np.random.random(embedding_dim))
        else:
            embd_new.append(embd[temp_word2id[id2w[i]]])

    vocab = vocab_new
    embd = embd_new

    vocab_size = len(vocab)

    embedding = np.asarray(embd)

    w = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=trainable, name="W-GloVe-Embd")
    embedding_placeholder = tf.placeholder(
        tf.float32, [vocab_size, embedding_dim])
    embedding_init = w.assign(embedding_placeholder)

    def initializer(sess):
        sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

    vocab = DEF_LS + vocab
    wu = get_variable('special_tokens',
                      shape=[len(DEF_LS), embedding_dim])
    wu = tf.concat([wu, w], axis=0)

    word2id = {vocab[ind]: ind for ind in range(0, len(vocab))}
    id2word = {i: vocab[i] for i in range(len(vocab))}

    return initializer, wu, word2id, id2word, embd


def load_partial_bert_layer(trainable=False, id2w=None, path='.'):
    embed_path = path + '\\bert\\bert.out'
    vocab_path = path + '\\bert\\vocab.txt'
    vocab = []
    embd = np.loadtxt(embed_path, dtype=np.float, delimiter=',')

    with open(vocab_path, encoding='utf-8') as f:
        for i in f:
            i = i.split('\n')[0]
            vocab.append(i)

    temp_word2id = {vocab[ind]: ind for ind in range(0, len(vocab))}
    embedding_dim = len(embd[0])
    if id2w is not None:
        vocab_new = []
        embd_new = []
        for i in range(4, args.vocab_size):
            vocab_new.append(id2w[i])
            if id2w[i] not in temp_word2id:
                embd_new.append(np.random.random(embedding_dim))
            else:
                embd_new.append(embd[temp_word2id[id2w[i]]])

        vocab = vocab_new
        embd = embd_new
    else:
        vocab_new = []
        embd_new = []
        for i in range(len(vocab)):
            if '[unused' in vocab[i] \
                    or (not isAn(vocab[i][0]) and vocab[i] not in string.punctuation):
                pass
            else:
                vocab_new.append(vocab[i])
                embd_new.append(embd[i])
        vocab = vocab_new
        embd = embd_new
        # print(vocab)
        print(len(vocab))

    vocab_size = len(vocab)

    embedding = np.asarray(embd)

    w = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=trainable, name="W-GloVe-Embd")
    embedding_placeholder = tf.placeholder(
        tf.float32, [vocab_size, embedding_dim])
    embedding_init = w.assign(embedding_placeholder)

    def initializer(sess):
        sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

    vocab = DEF_LS + vocab

    for i in DEF_LS:
        embd.insert(0, np.random.random(embedding_dim))

    wu = get_variable('special_tokens',
                          shape=[len(DEF_LS), embedding_dim])
    wu = tf.concat([wu, w], axis=0)

    word2id = {vocab[ind]: ind for ind in range(0, len(vocab))}
    id2word = {i: vocab[i] for i in range(len(vocab))}

    return initializer, wu, word2id, id2word, embd


def exponential_mechanism(pho, epsilon, delta):
    pho = np.array(pho)
    temp = np.exp(epsilon / (2 * delta) * pho)
    return temp / np.sum(temp)

def entropy(p):
    return np.mean(np.multiply(np.log(p), p))

def softmax(x, axis=None):
    x = np.array(x)
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

