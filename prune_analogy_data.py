from __future__ import division
from __future__ import print_function

import cPickle as pickle

class ListSink(object):
	def __init__(self):
		self.data = []
	def write(self, data):
		self.data.append(data)
	def __enter__(self): return self
	def __exit__(*x): pass


def conditional_context_manager(filename):
	if filename is not None:
		return open(filename, "w")
	else:
		return ListSink()


def get_valid_data(vocabulary, examples, destination=None):
	with conditional_context_manager(destination) as output_stream:
		for ex_index, example in enumerate(examples):
			if all([word in vocabulary for word in example]):
				output_stream.write(' '.join(example) + '\n' if destination is not None else example)
		if destination is None:
			return output_stream.data

def process_vocab_file(filename):
	with open(filename, 'r') as f:
		return set(pickle.load(f).values())

def process_examples_file(filename):
	with open(filename, 'r') as f:
		return [line.split() for line in f.readlines()]

if __name__ == '__main__':
	get_valid_data(process_vocab_file("data/reverse_dictionary.p"), process_examples_file("data/en_sem-para_SemRel.txt"), destination="data/analogies2.txt")