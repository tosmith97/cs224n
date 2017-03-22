# Utility file to consolidate various functions 

import xml.etree.ElementTree
import os, fnmatch
from process_xml import is_apostrophe_chunk, is_number, is_valid

def get_file_str(filename):
    root = xml.etree.ElementTree.parse(filename).getroot()

    # returns list of words for easier processing
    return get_word_sequence(root)


def get_sense_name(element):
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
    seq = [get_sense_name(word) for word in root]
    fixed_seq = []
    for i in range(len(seq) - 1):
        if is_apostrophe_chunk(seq[i]):
            continue
        to_append = seq[i]
        if is_apostrophe_chunk(seq[i+1]) and 'sense' not in seq[i]:
            to_append += seq[i+1]
        fixed_seq.append(to_append)
    if not is_apostrophe_chunk(seq[-1]) and len(seq[-1]) > 0:
        fixed_seq.append(seq[-1])

    fixed_seq = [word.lower() for word in fixed_seq if is_valid(word)]
    for i, s in enumerate(fixed_seq):
        if is_number(s):
            fixed_seq[i] = 'NUM'

    return fixed_seq


def get_dir_list(dirname, get_file_string):
    '''
    Recursively iterate through directories and compile
    all the file strings into one string
    '''
    full_string = []
    for dirpath, dirs, files in os.walk(dirname):
        for filename in fnmatch.filter(files, '*.xml'):
            full_string += get_file_string(os.path.join(dirpath, filename))
    return full_string


def get_mls_for_window(window, possible_senses):
    mls_window = []

    for word in window.split():
        senses = list(possible_senses[word])

        if len(senses) > 0:
            most_common_sense = senses[0]
        else:
            # note the difference for RNN!!
            most_common_sense = word
        mls_window.append(most_common_sense)

    return mls_window
    