
import xml.etree.ElementTree
import sys
import collections
import os
import fnmatch
import pickle

# adapting http://stackoverflow.com/questions/1912434/how-do-i-parse-xml-in-python (ElemlentTree)

# Constants
kWindowSize = 4

def main():
    '''
    Invocation: python process_xml.py <dirname>
    '''
    dirname = sys.argv[-1]
    with open('training_data.p', 'wb') as f:
        #all_data = parse_directory(dirname)
        full_string = get_dir_string(dirname)
        pickle.dump(full_string, f)

def parse_directory(dirname):
    '''
    dirname is a path to a directory
    consolidate all the file data
    into one list
    '''
    all_pairs = []
    for dirpath, dirs, files in os.walk(dirname):
        for filename in fnmatch.filter(files, '*.xml'):
            all_pairs += parse_file(os.path.join(dirpath, filename))
    return all_pairs
    

def parse_file(filename):
    '''
    Return skip prcoessing from given
    xml file
    '''
    root = xml.etree.ElementTree.parse(filename).getroot()
    return create_skipgram(get_word_sequence(root), kWindowSize)

def get_file_string(filename):
    '''
    Convert xml file to a single string,
    gluing together contractions (w/ apostrophes)
    '''
    root = xml.etree.ElementTree.parse(filename).getroot()
    return ' '.join(get_word_sequence(root))

def get_dir_string(dirname):
    '''
    Recursively iterate through directories and compile
    all the file strings into one string
    '''
    full_string = ''
    for dirpath, dirs, files in os.walk(dirname):
        for filename in fnmatch.filter(files, '*.xml'):
            full_string += get_file_string(os.path.join(dirpath, filename)) + ' '
    return full_string

def get_sense_name(element):
    '''
    tag a token with its sense
    '''
    suffix = element.get('sense')
    if suffix is None:
        suffix = ''
    return element.get('text') + suffix

def is_apostrophe_chunk(token):
    return len(token) > 0 and token[0] == '\''

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
    for i in xrange(len(seq) - 1):
        if is_apostrophe_chunk(seq[i]):
            continue
        to_append = seq[i]
        if is_apostrophe_chunk(seq[i+1]):
            to_append += seq[i+1]
        fixed_seq.append(to_append)
    if not is_apostrophe_chunk(seq[-1]) and len(seq[-1]) > 0:
        fixed_seq.append(seq[-1])
    return fixed_seq

        



def create_skipgram(word_sequence, window_size):
    '''
    Converts sequence of words to the following form:
    [(<center-word>, <context-word>), ...]
    consider splitting on sentences
    '''
    length = len(word_sequence)
    center_context_pairs = []
    for i in range(length):
        start = max(0, i - window_size)
        end = min(length, i + window_size + 1)
        context = word_sequence[start:i] + word_sequence[i+1:end]
        for ctxt in context:
            center_context_pairs.append((word_sequence[i], ctxt))
    return center_context_pairs

if __name__ == '__main__':
    main()