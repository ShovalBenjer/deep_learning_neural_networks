import keras
import keras.backend as K
from keras.layers import LSTM, Embedding, TimeDistributed, Input, Dense
from keras.models import Model
from tensorflow.python.client import device_lib

from tqdm import tqdm
import os, random
from argparse import ArgumentParser
import numpy as np

from tensorboardX import SummaryWriter

import util
from tensorflow.keras.losses import sparse_categorical_crossentropy

# Number of sample sentences to generate during training
CHECK = 5

def generate_seq(model: Model, seed, size, temperature=1.0):
    """
    Generate a sequence of word indices from the trained model.

    :param model: The complete RNN language model.
    :param seed: A numpy array (1D) of word indices to start the generation.
    :param size: The total length of the sequence to generate.
    :param temperature: Controls randomness in sampling. For temperature=1.0, sample directly
                        according to model probabilities. Lower temperatures yield more predictable
                        output; higher temperatures yield more diverse output. For temperature=0.0 the
                        generation is greedy.
    :return: A list of integers representing the generated sequence.
    """
    ls = seed.shape[0]
    # Concatenate the seed with zeros (for remaining positions)
    tokens = np.concatenate([seed, np.zeros(size - ls)])
    for i in range(ls, size):
        probs = model.predict(tokens[None, :])
        next_token = util.sample_logits(probs[0, i-1, :], temperature=temperature)
        tokens[i] = next_token
    return [int(t) for t in tokens]

def sparse_loss(y_true, y_pred):
    """
    Compute sparse categorical crossentropy loss using TensorFlow Keras's loss function.
    """
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def build_model(numwords, lstm_capacity, extra, reverse):
    """
    Build and return a Keras Model for language modeling.
    
    :param numwords: Size of the vocabulary.
    :param lstm_capacity: Dimensionality of the LSTM hidden state.
    :param extra: Number of extra LSTM layers (None means only one layer).
    :param reverse: Boolean flag; if True, LSTM layers use go_backwards=True.
    :return: A compiled Keras model.
    """
    inp = Input(shape=(None,))
    embed = Embedding(numwords, lstm_capacity)
    x = embed(inp)
    # First LSTM layer with reverse flag
    x = LSTM(lstm_capacity, return_sequences=True, go_backwards=reverse)(x)
    # Extra LSTM layers if specified
    if extra is not None:
        for _ in range(extra):
            x = LSTM(lstm_capacity, return_sequences=True, go_backwards=reverse)(x)
    dense = Dense(numwords, activation='linear')
    out = TimeDistributed(dense)(x)
    model = Model(inp, out)
    return model

def train_model(options, train_data, val_data, test_data, w2i, i2w):
    """
    Build and train a language model using globally padded data.
    Expects train_data, val_data, and test_data to be numpy arrays of shape [num_sentences, seq_length].
    
    The training loop:
      - Shuffles the training data each epoch.
      - Iterates over mini-batches.
      - Logs training loss via TensorBoardX.
      - After training, computes perplexity on train, validation, and test sets.
      - Generates sample sentences and computes sentence probabilities.
    
    :param options: An options object with training hyperparameters.
    :param train_data, val_data, test_data: Numpy arrays (globally padded) of shape [N, L].
    :param w2i, i2w: Word-to-index and index-to-word dictionaries.
    :return: The trained Keras model.
    """
    writer = SummaryWriter(log_dir=options.tb_dir)
    np.random.seed(options.seed)
    
    numwords = len(i2w)
    model = build_model(numwords, options.lstm_capacity, options.extra, options.reverse)
    opt = keras.optimizers.Adam(learning_rate=options.lr)
    model.compile(opt, sparse_loss)
    model.summary()
    
    num_train = train_data.shape[0]
    instances_seen = 0
    for epoch in range(options.epochs):
        # Shuffle training data each epoch
        indices = np.arange(num_train)
        np.random.shuffle(indices)
        train_data = train_data[indices]
        for i in tqdm(range(0, num_train, options.batch)):
            batch = train_data[i:i+options.batch]
            n, l = batch.shape
            # Prepend start symbol (assumed to be index 1) and append pad symbol (assumed index 0)
            batch_in = np.concatenate([np.ones((n, 1), dtype='int32'), batch], axis=1)
            batch_out = np.concatenate([batch, np.zeros((n, 1), dtype='int32')], axis=1)
            loss = model.train_on_batch(batch_in, batch_out[:, :, None])
            instances_seen += n
            writer.add_scalar('lm/train_batch_loss', float(loss), instances_seen)
        print("Epoch {} complete".format(epoch+1))
        
        # Generate sample sentences at various temperatures.
        for temp in [0.0, 0.9, 1.0, 1.1, 1.2]:
            print("### TEMP", temp)
            for _ in range(CHECK):
                # Here, train_data is a 2D numpy array (each row is one padded sentence)
                idx = random.randint(0, num_train - 1)
                sentence = train_data[idx]  # sentence is a 1D array
                if len(sentence) > 20:
                    seed = sentence[:20]
                else:
                    seed = sentence
                seed = np.insert(seed, 0, 1)  # Prepend <START> token (assumed index 1)
                gen = generate_seq(model, seed, 60, temperature=temp)
                def decode(seq):
                    return ' '.join(i2w[str(i)] for i in seq)
                print('*** [', decode(seed), '] ', decode(gen[len(seed):]))
    writer.close()
    
    # Compute perplexity on each dataset
    ppl_train = compute_perplexity(model, train_data, options.batch)
    ppl_val = compute_perplexity(model, val_data, options.batch)
    ppl_test = compute_perplexity(model, test_data, options.batch)
    
    print("Perplexity (Train): {:.2f}".format(ppl_train))
    print("Perplexity (Validation): {:.2f}".format(ppl_val))
    print("Perplexity (Test): {:.2f}".format(ppl_test))
    
    # Generate sentence of length 7 starting with "love I" at different temperatures
    seed_words = "love I".split()
    seed_ids = [w2i.get(word.lower(), w2i.get('<UNK>')) for word in seed_words]
    seed_ids = np.array(seed_ids)
    print("\nSentence generation (length=7) starting with 'love I':")
    for temp in [0.1, 1.0, 10.0]:
        gen = generate_seq(model, np.insert(seed_ids, 0, 1), size=7, temperature=temp)
        def decode(seq):
            return ' '.join(i2w[str(i)] for i in seq)
        print("Temperature {}: {}".format(temp, decode(gen)))
    
    # Compute probability for the generated sentence and for "love i cupcakes"
    def decode(seq):
        return ' '.join(i2w[str(i)] for i in seq)
    generated_sentence = decode(gen)
    p, logp = sentence_probability(generated_sentence, model, w2i, i2w)
    print("\nProbability for generated sentence '{}': {:.2e} (log={:.2f})".format(generated_sentence, p, logp))
    sentence2 = "love i cupcakes"
    p2, logp2 = sentence_probability(sentence2, model, w2i, i2w)
    print("Probability for sentence 'love i cupcakes': {:.2e} (log={:.2f})".format(p2, logp2))
    
    return model

def sentence_probability(sentence, model, w2i, i2w):
    """
    Compute the probability and log-probability of a given sentence.
    
    The sentence is tokenized by spaces; unknown words are replaced by <UNK>.
    The model is assumed to predict the next word given the previous tokens.
    
    :return: Tuple (probability, log_probability)
    """
    words_in = sentence.strip().split()
    unk = w2i.get('<UNK>', 2)
    token_ids = [w2i.get(word.lower(), unk) for word in words_in]
    token_ids = [1] + token_ids  # Prepend <START> token (assumed index 1)
    token_ids = np.array(token_ids)[None, :]
    logits = model.predict(token_ids)
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    p = 1.0
    logp = 0.0
    for t in range(1, token_ids.shape[1]):
        prob = probs[0, t-1, token_ids[0, t]]
        p *= prob
        logp += np.log(prob + 1e-10)
    return p, logp

if __name__ == "__main__":
    ## Parse command-line options
    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=20, type=int)
    parser.add_argument("-E", "--embedding-size",
                        dest="embedding_size",
                        help="Size of the word embeddings on the input layer.",
                        default=300, type=int)
    parser.add_argument("-o", "--output-every",
                        dest="out_every",
                        help="Output every n epochs.",
                        default=1, type=int)
    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)
    parser.add_argument("-b", "--batch-size",
                        dest="batch",
                        help="Batch size",
                        default=128, type=int)
    parser.add_argument("-t", "--task",
                        dest="task",
                        help="Task",
                        default='wikisimple', type=str)
    parser.add_argument("-D", "--data-directory",
                        dest="data_dir",
                        help="Data file directory (one sentence per line).",
                        default='./data', type=str)
    parser.add_argument("-L", "--lstm-hidden-size",
                        dest="lstm_capacity",
                        help="LSTM capacity",
                        default=256, type=int)
    parser.add_argument("-m", "--max_length",
                        dest="max_length",
                        help="Max length",
                        default=None, type=int)
    parser.add_argument("-w", "--top_words",
                        dest="top_words",
                        help="Top words (vocab size)",
                        default=10000, type=int)
    parser.add_argument("-I", "--limit",
                        dest="limit",
                        help="Character cap for the corpus",
                        default=None, type=int)
    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="TensorBoard log directory",
                        default='./runs/words', type=str)
    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random (seed is printed for reproducibility).",
                        default=-1, type=int)
    parser.add_argument("-x", "--extra-layers",
                        dest="extra",
                        help="Number of extra LSTM layers (None means one LSTM layer only).",
                        default=None, type=int)
    # New flag to enable reverse training (LSTM go_backwards)
    parser.add_argument("-R", "--reverse",
                        dest="reverse",
                        action="store_true",
                        help="If set, train the model with reversed sequences (go_backwards=True in LSTM).")
    options = parser.parse_args()
    print('OPTIONS', options)
    go(options)
