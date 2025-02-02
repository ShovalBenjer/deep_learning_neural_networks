from keras.preprocessing.text import text_to_word_sequence
from keras.layers import Layer
import keras.utils
import keras.backend as K

from nltk import FreqDist
import numpy as np

from keras.preprocessing import sequence

from scipy.misc import logsumexp

from collections import defaultdict, Counter, OrderedDict

import os

"""
Various utility functions for loading data and performing common operations.
Some of this code is based on https://github.com/ChunML/seq2seq/blob/master/seq2seq_utils.py
"""

# Special tokens
EXTRA_SYMBOLS = ['<PAD>', '<START>', '<UNK>', '<EOS>']
DIR = os.path.dirname(os.path.realpath(__file__))

def load_words(source, vocab_size=10000, limit=None, max_length=None):
    """
    Loads sentences (or other natural language sequences) from a text file. Assumes one sequence per line.
    
    :param source: Path to the text file.
    :param vocab_size: Maximum number of words to retain. If there are more unique words than this,
                       only the most frequent (vocab_size - len(EXTRA_SYMBOLS)) words are used;
                       the rest are replaced by the <UNK> symbol.
    :param limit: If not None, only the first "limit" characters are read (useful for debugging).
    :param max_length: If not None, sentences longer than this number of words are discarded.
    :return: A tuple containing:
             (1) A list of lists of integers (encoded sentences),
             (2) A dictionary mapping words to indices,
             (3) A list mapping indices to words.
    """
    # Read the raw text
    with open(source, 'r') as f:
        x_data = f.read()
    print('raw data read')
    
    if limit is not None:
        x_data = x_data[:limit]
    
    # Split text into sequences (one per line)
    x = [text_to_word_sequence(line) for line in x_data.split('\n') if len(line) > 0]
    
    if max_length is not None:
        x = [s for s in x if len(s) <= max_length]
    
    # Build vocabulary: count word frequencies
    dist = FreqDist(np.hstack(x))
    x_vocab = dist.most_common(vocab_size - len(EXTRA_SYMBOLS))
    
    # Create index-to-word mapping; prepend special tokens
    i2w = [word[0] for word in x_vocab]
    i2w = EXTRA_SYMBOLS + i2w
    
    # Create word-to-index mapping
    w2i = {word: ix for ix, word in enumerate(i2w)}
    
    # Encode each word in each sentence as its index
    for i, sentence in enumerate(x):
        for j, word in enumerate(sentence):
            if word in w2i:
                x[i][j] = w2i[word]
            else:
                x[i][j] = w2i['<UNK>']
    
    return x, w2i, i2w

def load_characters(source, length=None, limit=None):
    """
    Reads a text file as a stream of characters and splits it into chunks.
    
    :param source: Path to the text file.
    :param length: Size of each chunk. If None, the text is split by line.
    :param limit: If not None, only the first "limit" characters are read.
    :return: A tuple containing:
             (1) A list of lists (each list is a sequence of characters),
             (2) A dictionary mapping characters to indices,
             (3) A list mapping indices to characters.
    """
    with open(source, 'r') as f:
        x_data = f.read()
    print('raw data read')
    
    if limit is not None:
        x_data = x_data[:limit]
    
    if length is None:
        x = [list(line) for line in x_data.split('\n') if len(line) > 0]
    else:
        x = [list(chunk) for chunk in chunks(x_data, length)]
    
    # Build vocabulary of characters
    chars = set()
    for line in x:
        for char in line:
            chars.add(char)
    
    i2c = list(chars)
    i2c = EXTRA_SYMBOLS + i2c
    c2i = {char: ix for ix, char in enumerate(i2c)}
    
    for i, sentence in enumerate(x):
        for j, char in enumerate(sentence):
            if char in c2i:
                x[i][j] = c2i[char]
            else:
                x[i][j] = c2i['<UNK>']
    
    return x, c2i, i2c

def process_data(word_sentences, max_len, word_to_ix):
    """
    Vectorizes a list of encoded sentences into one-hot vectors.
    
    :param word_sentences: List of sentences (each is a list of integer word indices).
    :param max_len: Maximum sentence length.
    :param word_to_ix: Dictionary mapping words to indices.
    :return: A numpy array of shape (num_sentences, max_len, vocab_size).
    """
    sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
    for i, sentence in enumerate(word_sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1.0
    return sequences

def batch_pad(x, batch_size, min_length=3, add_eos=False, extra_padding=0):
    """
    Sorts and pads a list of integer sequences into batches so that every sentence in a batch
    has the same length.
    
    :param x: List of integer sequences.
    :param batch_size: Number of sequences per batch.
    :param min_length: Minimum length of sequences to be considered.
    :param add_eos: If True, appends the <EOS> token to each sentence.
    :param extra_padding: Additional padding to add.
    :return: A list of numpy arrays (each array is a padded batch).
    """
    x = sorted(x, key=lambda l: len(l))
    
    if add_eos:
        eos = EXTRA_SYMBOLS.index('<EOS>')
        x = [sent + [eos] for sent in x]
    
    batches = []
    start = 0
    while start < len(x):
        end = start + batch_size
        if end > len(x):
            end = len(x)
        batch = x[start:end]
        mlen = max([len(l) + extra_padding for l in batch])
        if mlen >= min_length:
            batch = sequence.pad_sequences(batch, maxlen=mlen, dtype='int32', padding='post', truncating='post')
            batches.append(batch)
        start += batch_size

    print('max length per batch: ', [max([len(l) for l in batch]) for batch in batches])
    return batches

def to_categorical(batch, num_classes):
    """
    Converts a batch of padded integer sequences into one-hot encoded format.
    
    :param batch: A numpy array of shape (batch_size, sequence_length).
    :param num_classes: Total number of classes (vocabulary size).
    :return: A numpy array of shape (batch_size, sequence_length, num_classes).
    """
    b, l = batch.shape
    out = np.zeros((b, l, num_classes))
    for i in range(b):
        seq = batch[i, :]
        out[i, :, :] = keras.utils.to_categorical(seq, num_classes=num_classes)
    return out

def chunks(l, n):
    """Yield successive n-sized chunks from list l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def sample(preds, temperature=1.0):
    """
    Sample an index from a probability vector.
    
    :param preds: Array of probabilities.
    :param temperature: Sampling temperature.
    :return: Selected index.
    """
    preds = np.asarray(preds).astype('float64')
    if temperature == 0.0:
        return np.argmax(preds)
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def sample_logits(preds, temperature=1.0):
    """
    Sample an index from a logit vector.
    
    :param preds: Array of logits.
    :param temperature: Sampling temperature.
    :return: Selected index.
    """
    preds = np.asarray(preds).astype('float64')
    if temperature == 0.0:
        return np.argmax(preds)
    preds = preds / temperature
    preds = preds - logsumexp(preds)
    choice = np.random.choice(len(preds), 1, p=np.exp(preds))
    return choice

class KLLayer(Layer):
    """
    A custom Keras layer that performs an identity transformation while adding
    a KL divergence loss to the final model loss.
    
    To scale the KL loss term, call:
         K.set_value(kl_layer.weight, new_value)
    """
    def __init__(self, weight=None, *args, **kwargs):
        self.is_placeholder = True
        self.weight = weight
        super().__init__(*args, **kwargs)
    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        loss = K.mean(kl_batch)
        if self.weight is not None:
            loss = loss * self.weight
        self.add_loss(loss, inputs=inputs)
        return inputs

class Sample(Layer):
    """
    A custom Keras layer that performs the sampling step in a variational autoencoder.
    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super().__init__(*args, **kwargs)
    def call(self, inputs):
        mu, log_var, eps = inputs
        z = K.exp(0.5 * log_var) * eps + mu
        return z
    def compute_output_shape(self, input_shape):
        shape_mu, _, _ = input_shape
        return shape_mu

def interpolate(start, end, steps):
    """
    Linearly interpolates between two vectors.
    
    :param start: Starting vector.
    :param end: Ending vector.
    :param steps: Number of interpolation steps.
    :return: A numpy array with interpolated vectors.
    """
    result = np.zeros((steps + 2, start.shape[0]))
    for i, d in enumerate(np.linspace(0, 1, steps + 2)):
        result[i, :] = start * (1 - d) + end * d
    return result

class OrderedCounter(Counter, OrderedDict):
    """
    A Counter that remembers the order in which elements were first encountered.
    """
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))
    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

def idx2word(idx, i2w, pad_idx):
    """
    Converts sequences of indices into human-readable sentences.
    
    :param idx: A list of sequences (each sequence is a list of indices).
    :param i2w: Dictionary mapping indices to words.
    :param pad_idx: The padding index (when encountered, stop the sentence).
    :return: A list of decoded sentences.
    """
    sent_str = [str() for _ in range(len(idx))]
    for i, sent in enumerate(idx):
        for word_id in sent:
            if word_id == pad_idx:
                break
            sent_str[i] += i2w[str(word_id.item())] + " "
        sent_str[i] = sent_str[i].strip()
    return sent_str
