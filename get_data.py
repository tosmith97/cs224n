# Adopted from lecture

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
import os
import random
import cPickle as pickle
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from tqdm import tqdm

from defs import VOCAB_SIZE


# Read the data into a list of strings.
def read_data():
  """Open pickle from process_xml.py"""
  with open('training_data.p', 'rb') as f:
    data = pickle.load(f).split()
  return data

words = read_data()
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = VOCAB_SIZE

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    # word along with 'ranking' in terms of popularity
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      # 'rank'
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.

data_index = 0
# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

batch_size = 128
skip_window = 2       # How many words to consider left and right.
num_skips = 1         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# Step 5: Begin training.
num_steps = 100000
training_data = []
for step in tqdm(range(num_steps)):
  batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
  training_data.append((batch_inputs, batch_labels))

data_folder = './data'
if not os.path.exists(data_folder):
  os.makedirs(data_folder)
print('Saving training data...')
pickle.dump(training_data, open( "./data/train.p", "wb" ))
print('Saving validation data...')
pickle.dump(valid_examples, open("./data/val.p", "wb"))
print('Saving reverse_dictionary')
pickle.dump(reverse_dictionary, open("./data/rd2.p", "wb"))  
