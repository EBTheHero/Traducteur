# -*- coding: utf-8 -*-


from __future__ import print_function
from keras.models import Sequential
from keras import layers
import keras
import numpy as np
from six.moves import range
import importeurdonnees
import h5py
import os


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C.

        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)


# Parameters for the model and dataset.
TRAINING_SIZE = 100000

# Name of the file where the model is saved
SAVEFILE = "model.h5"

# if true, load previous model and continue training
LOAD = os.path.isfile(SAVEFILE)

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.
MAXLEN = 41

# All the numbers, plus sign and space for padding.
chars = r'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"{}:;, '
ctable = CharacterTable(chars)

questions = []
expected = []

print('Importing data...')

DataIn = importeurdonnees.getin(TRAINING_SIZE)
DataOut = importeurdonnees.getout(TRAINING_SIZE)

print("Parsing data")

for i in DataIn:

    # Pad the data with spaces such that it is always MAXLEN.
    q = i
    query = q + ' ' * (MAXLEN - len(q))
    questions.append(query)

for i in DataOut:
    ans = i
    # Answers can be of maximum size MAXLEN.
    ans += ' ' * (MAXLEN - len(ans))
    expected.append(ans)


print('Total addition questions:', len(questions))

print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, MAXLEN)

# Shuffle (x, y) in unison as the later parts of x will almost all be larger
# digits.
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

# https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
BATCH_SIZE = 32

if LOAD:
    print("Loading model...")
    model = keras.models.load_model(SAVEFILE)
else:
    RNN = layers.LSTM
    HIDDEN_SIZE = 1000

    LAYERS = 1

    print('Build model...')
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
    # Note: In a situation where your input sequences have a variable length,
    # use input_shape=(None, num_feature).
    model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
    # As the decoder RNN's input, repeatedly provide with the last hidden state of
    # RNN for each time step. Repeat 'MAXLEN' times as that's the maximum
    # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
    model.add(layers.RepeatVector(MAXLEN))
    # The decoder RNN could be multiple layers stacked or a single layer.
    for _ in range(LAYERS):
        # By setting return_sequences to True, return not only the last output but
        # all the outputs so far in the form of (num_samples, timesteps,
        # output_dim). This is necessary as TimeDistributed in the below expects
        # the first dimension to be the timesteps.
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.

    opt = keras.optimizers.adam(lr=0.0001)

    model.add(layers.TimeDistributed(layers.Dense(len(chars))))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'],
                  )
    model.summary()


# Train the model each generation and show predictions against the validation
# dataset.
for iteration in range(1, 2000):
    print()
    print('-' * 50)
    print('Iteration', iteration, ", Batch size:", BATCH_SIZE)
    thing = model.fit(x_train, y_train,
                      batch_size=BATCH_SIZE,
                      epochs=1,
                      validation_data=(x_val, y_val))

    if thing.history['val_acc'][0] < 0.7:
        BATCH_SIZE = 32

    if thing.history['val_acc'][0] > 0.7:
        BATCH_SIZE = 128

    if thing.history['val_acc'][0] > 0.8:
        BATCH_SIZE = 300
    if thing.history['val_acc'][0] > 0.9:
        BATCH_SIZE = 400

    model.save(SAVEFILE)

    if thing.history['val_acc'][0] >= 1.0:
        break
    # Select 10 samples from the validation set at random so we can visualize
    # errors.
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('In:', q, end=' ')
        print('Expected:', correct, end=' ')
        print('Out:', end=' ')
        print(guess)


model.save(SAVEFILE)
input("FRICKIN' DONE! saved model.h5")
