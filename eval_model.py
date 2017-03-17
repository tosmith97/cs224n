import xml.etree.ElementTree
import sys
import collections
import os
import fnmatch
import cPickle as pickle

from process_xml import get_dir_list, is_valid, is_apostrophe_chunk, get_sense_name, is_number
import predict_sense
from defs import WINDOW_SIZE


def main():
    dirname = sys.argv[-1]
    eval_data = get_dir_list(dirname, get_file_str)
    idx_of_sense = collections.defaultdict()

    # only put indices of words w/ senses in defaultdict to minimize space complexity
    # want to pass words without senses to eval
    for i, w in enumerate(eval_data):
        if '/' in w:
            eval_data[i] = w.split('/', 1)[0]
            idx_of_sense[i] = w
    
    see_prediction_results(eval_data, idx_of_sense)

    # print(eval_data)
    # print(idx_of_sense)


def get_file_str(filename):
    root = xml.etree.ElementTree.parse(filename).getroot()

    # returns list of words for easier processing
    return get_word_sequence(root)


def process_element(element):
    '''
    tag a token with its sense
    '''
    suffix = element.get('sense')
    if suffix is None:
        suffix = ''
    return element.get('text') + suffix


def get_word_sequence(root):
    '''
    Iterate through xml root 
    and compile everything into
    a list of strings. Handle apostrophes
    by gluing together to form 
    contractions.
    '''
    seq = [process_element(word) for word in root]
    fixed_seq = []
    for i in xrange(len(seq) - 1):
        if is_apostrophe_chunk(seq[i]):
            continue
        to_append = seq[i]
        if is_apostrophe_chunk(seq[i+1]):
            to_append += seq[i+1]
        fixed_seq.append(to_append)
    if not is_apostrophe_chunk(seq[-1]) and len(seq[-1]) > 0:
        fixed_seq.append(seq[-1])

    fixed_seq = [word.lower() for word in fixed_seq if is_valid(word)]
    for i, s in enumerate(fixed_seq):
        if is_number(s):
            fixed_seq[i] = 'NUM'
        
    return fixed_seq


def window_has_senses(idx_of_sense, idx_range):
    # idx_range is tuple, idx_of_sense is dict w/ all the stuff
    return any(key in idx_of_sense for key in range(idx_range[0], idx_range[1]))


def get_senses_in_window(window, idx_of_sense, idx_range, eval_data):
    window_senses = []
    
    for idx in range(idx_range[0], idx_range[1]):
        if idx in idx_of_sense:
            window_senses.append(idx_of_sense[idx])
        else:
            window_senses.append(eval_data[idx])
    
    return window_senses



def see_prediction_results(eval_data, idx_of_sense):
    # with open("data/reverse_dictionary.p", "rb") as f:
    #     rd = pickle.load(f)

    # vocab = read_file("data/vocab.txt")
    # word_vectors = read_file("data/word_vectors.txt")
    # si, ps = generate_dicts(rd)
    # embeddings = load_embeddings(vocab, word_vectors, si)

    # take window
    # See if words in window have senses that need predicting 
    # predict senses for WINDOW_SIZE
    # compare predicted senses to actual senses
    # have acc = num_correct/tot
    
    num_correct = 0
    tot_comparisons = 0
    incorrect = []
    correct = []

    embeddings = predict_sense.embeddings
    string_to_index = predict_sense.si
    possible_senses = predict_sense.ps


    # Current predict_sense finds predictions for whole window - do we need a center word? 
    for i in range(WINDOW_SIZE, len(eval_data) - WINDOW_SIZE):
        if window_has_senses(idx_of_sense, (i - WINDOW_SIZE, i + WINDOW_SIZE + 1)):
            # predict that shit
            window = ' '.join(eval_data[i - WINDOW_SIZE: i + WINDOW_SIZE + 1])
            
            # if we don't tag a predicted word with a sense, 
            # despite it having senses, tag it with the most frequent sense.            
            pred_senses = list(predict_sense.predict_sense(window, string_to_index, possible_senses, embeddings))
            for k, p_s in enumerate(pred_senses):
                check = window.split()[k]
                num_senses = len(possible_senses[p_s])

                # if a given word has >1 sense and the sense it's tagged with is smallest in that list
                if num_senses > 1 and p_s == min((word for word in possible_senses[check]), key=len): 
                    pred_senses[k] = list(possible_senses[p_s])[0] if (pred_senses[k] != list(possible_senses[p_s])[0]) else list(possible_senses[p_s])[1] 

            senses = get_senses_in_window(window, idx_of_sense, (i - WINDOW_SIZE, i + WINDOW_SIZE + 1), eval_data)

            for word in xrange(len(pred_senses) - 1):
                tot_comparisons += 1
                if pred_senses[word] == senses[word]:
                    num_correct += 1
                    correct.append(("Predicted: " + pred_senses[word], "Actual:" + senses[word]))
                else:
                    if '/' in senses[word]:
                        incorrect.append(("Predicted: " + pred_senses[word], "Actual:" + senses[word]))
                    else:
                        tot_comparisons -= 1
        
            if i % 10 == 0 and tot_comparisons > 0:
                print("Writing to file...")
                with open("incorrect.txt", 'w') as f:
                    f.write("Tested: " + str(tot_comparisons))
                    f.write("Accuracy:" + str(float(num_correct)/tot_comparisons))
                    f.write('\n'.join('%s %s' % x for x in incorrect))    

    print("Accuracy:", float(num_correct)/tot_comparisons)
    with open("incorrect.txt", 'w') as f:
        f.write('\n'.join('%s %s' % x for x in incorrect))




    # Probs:
    #   words will be counted 4x?
    #   What if miscalc one time but not another?
    #   Need to pad windows somehow for beginning/end





if __name__ == '__main__':
    main()


# compare our prediction to actual sense for a given word
# probs wanna check that word has sense to begin with


