from __future__ import division
from __future__ import print_function

import cPickle as pickle

import numpy as np
import tensorflow as tf


class ListSink(object):
    def __init__(self):
        self.data = []
    def write(self, data):
        self.data.append(data)
    def __enter__(self): return self
    def __exit__(*x): pass


class Utilities(object):
    @staticmethod
    def conditional_context_manager(filename):
        if filename is not None:
            return open(filename, "w")
        else:
            return ListSink()


class DataParser(object):
    def __init__(self, reverse_dict_file=None, word_embs_file=None, analogies_file=None):
        print("Parsing data.")
        print("Parsing word embedding data...")
        self.word_embs = self.get_word_embs(word_embs_file)
        print("Parsing reverse dictionary, vocab, and mapping data...")
        self.reverse_dict = self.get_reverse_dict(reverse_dict_file)
        if len(self.reverse_dict) != self.word_embs.shape[0]:
            self.reverse_dict = {k: v for k, v in self.reverse_dict.items() if k < self.word_embs.shape[0]}
        self.vocab = self.get_vocab(self.reverse_dict)
        self.sense_to_idx_mapping = self.get_word_idx_mapping(self.reverse_dict)
        self.words_only_reverse_dict = self.get_words_only_reverse_dict(self.reverse_dict)
        
        print("Parsing analogy data...")
        self.analogies = self.get_valid_analogies(self.vocab,self.get_analogies(analogies_file))
        self.analogies_idxs = self.get_analogies_as_idxs(self.analogies)
        

        print("Done parsing data!")

    def get_words_only_reverse_dict(self, reverse_dict=None):
        if reverse_dict is None: return None
        return {k: v.split("/")[0] for k, v in reverse_dict.items()}

    def get_analogies_as_idxs(self, analogies=None):
        if analogies is None: return None
        analogies_idxs = [[self.sense_to_idx_mapping[word] for word in analogy] for analogy in analogies]
        return np.asarray(analogies_idxs, dtype=np.int32)

    def get_valid_analogies(self, vocabulary=None, examples=None, destination=None):
        if vocabulary is None or examples is None: return None
        with Utilities.conditional_context_manager(destination) as output_stream:
            for ex_index, example in enumerate(examples):
                if all([word in vocabulary for word in example]):
                    output_stream.write(' '.join(example) + '\n' if destination is not None else example)
            if destination is None:
                return output_stream.data

    def get_reverse_dict(self, filename=None):
        if filename is None: return None
        with open(filename, 'r') as f:
            return pickle.load(f)

    def get_vocab(self, reverse_dict=None):
        if reverse_dict is None: return None
        return set(reverse_dict.values())

    def get_word_idx_mapping(self, reverse_dict=None):
        if reverse_dict is None: return None
        return {v: k for k, v in reverse_dict.items()}

    def get_analogies(self, filename=None):
        if filename is None: return None
        with open(filename, 'r') as f:
            return [line.split() for line in f.readlines()]

    def get_word_embs(self, filename=None):
        if filename is None: return None
        return np.loadtxt(filename)


class Config(object):
    num_best = 4
    batch_size = 2500
    

class AnalogyEvaluation(object):
    def __init__(self, config, embeddings, analogies, session, reverse_dict):
        self.config = config
        self.embeddings = embeddings
        self.analogies = analogies
        self._session = session
        self.reverse_dict = reverse_dict

    def build_eval_graph(self):
        word_idxs = [tf.placeholder(dtype=tf.int32) for _ in xrange(3)]
        embeddings = tf.placeholder(dtype=tf.float32)
        norm_embeddings = tf.nn.l2_normalize(embeddings, 1)
        word_embs = [tf.gather(norm_embeddings, word_idxs[i]) for i in xrange(3)]
        target_emb = word_embs[2] + (word_embs[1] - word_embs[0])
        cosine_sims = tf.matmul(target_emb, norm_embeddings, transpose_b=True)
        _, pred_idxs = tf.nn.top_k(cosine_sims, self.config.num_best)

        self._analogy_word_placeholders = word_idxs
        self._analogy_pred_idxs = pred_idxs
        self._embeddings = norm_embeddings

    def predict(self, analogies):
        feed_dict = {self._analogy_word_placeholders[i]: analogies[:, i] for i in xrange(3)}
        feed_dict[self._embeddings] = self.embeddings
        pred_idxs = self._session.run([self._analogy_pred_idxs], feed_dict)
        return pred_idxs

    def eval(self):
        correct = 0
        correct_on_first = 0
        total = self.analogies.shape[0]
    
        start = 0
        for i in xrange(0, total, self.config.batch_size):
            analogy_batch = self.analogies[i:i + self.config.batch_size, :]
            print("Processing analogies {0}-{1}...".format(i, i + self.config.batch_size))
            pred_idxs = self.predict(analogy_batch)[0]
            for analogy_idx in xrange(analogy_batch.shape[0]):
                info = "Analogy = {0} : {1} :: {2} : {3}. Guesses = ".format(*[self.reverse_dict[i] for i in analogy_batch[analogy_idx]])
                for j in xrange(self.config.num_best):
                    info += "{0},".format(str(self.reverse_dict[pred_idxs[analogy_idx, j]]))
                    if self.reverse_dict[pred_idxs[analogy_idx, j]] == self.reverse_dict[analogy_batch[analogy_idx, 3]]:
                        if j == 0:
                            correct_on_first += 1
                        correct += 1
                        break
                print(info)
        
        print("Finished evaluating analogies. Accuracy = {0}/{1} = {2}".format(correct, total, correct * 100.0 / total))

if __name__ == "__main__":
    data = DataParser(reverse_dict_file="data/reverse_dictionary.p", word_embs_file="data/word_vectors.txt", analogies_file="data/en_sem-para_SemRel.txt")
    with tf.Graph().as_default(), tf.Session() as session:
        analogy_eval = AnalogyEvaluation(config=Config(), embeddings=data.word_embs, analogies=data.analogies_idxs, session=session, reverse_dict=data.words_only_reverse_dict)
        analogy_eval.build_eval_graph()
        analogy_eval.eval()