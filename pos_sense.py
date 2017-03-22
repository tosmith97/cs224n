import cPickle as pickle
import collections

all_words = []
with open('data/senses/input.txt', 'r') as f:
    for line in f:
        all_words += line.split()

occurence_counts = collections.Counter(all_words)
possible_senses = collections.defaultdict(set) 
# possible senses maps a word to a set of 
# tuples (<word>, <num times it occurs>)
for w in occurence_counts:
    no_tag = w.split('/')[0]
    sense_occ = (w, occurence_counts[w])
    possible_senses[no_tag].add(sense_occ)

with open('possible_senses.p', 'wb') as f:
    pickle.dump(possible_senses, f)


