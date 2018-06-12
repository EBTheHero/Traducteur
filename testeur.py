# -*- coding: utf-8 -*-
'''An implementation of sequence to sequence learning for performing addition

Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)

Input may optionally be reversed, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.

Two digits reversed:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs

Three digits reversed:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs

Four digits reversed:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs

Five digits reversed:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs
'''

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


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# Parameters for the model and dataset.
TRAINING_SIZE = 100000

# if true, load previous model and continue training
LOAD = True

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.
MAXLEN = 63

# All the numbers, plus sign and space for padding.
chars = r'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"{}:;, '
ctable = CharacterTable(chars)

questions = []
expected = []

print('Importing data...')

DataIn = ["GEBS;RGOR;PAWD", "PFWM;WSMF;PFKW;WMDN"]

print("Parsing data")

for i in DataIn:
    
    # Pad the data with spaces such that it is always MAXLEN.
    q = i
    query = q + ' ' * (MAXLEN - len(q))
    questions.append(query)


#while len(questions) < TRAINING_SIZE:
#    f = lambda: str(''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
#                    for i in range(MAXLEN - 2)))
#    a = f()
#    # Skip any addition questions we've already seen
#    # Also skip any such that x+Y == Y+x (hence the sorting).
#    key = a
#    if key in seen:
#        continue
#    seen.add(key)
#    # Pad the data with spaces such that it is always MAXLEN.
#    q = a
#    query = q + ' ' * (MAXLEN - len(q))
#    ans = '*{}*'.format(a)
#    # Answers can be of maximum size MAXLEN.
#    ans += ' ' * (MAXLEN - len(ans))
#    questions.append(query)
#    expected.append(ans)
print('Total addition questions:', len(questions))

print('Vectorization...')
x_guess = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x_guess[i] = ctable.encode(sentence, MAXLEN)

# Shuffle (x, y) in unison as the later parts of x will almost all be larger
# digits.

# Try replacing GRU, or SimpleRNN.

BATCH_SIZE = 64


print("Loading model...")
model = keras.models.load_model("model.h5")


# Train the model each generation and show predictions against the validation
# dataset.

for i in range(0,len(x_guess)):
        rowx = x_guess[np.array([i])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q, end=' ')
        print(guess)


#for i in range(len(x_guess) - 1):
#    rowx = x_guess[i]
#    preds = model.predict_classes(rowx, verbose=0)
#    q = ctable.decode(rowx[0])
#    guess = ctable.decode(preds[0], calc_argmax=False)
#    print('Q', q, end=' ')
#    
#    print('->' ,guess, end=' ')

input("done guessing")