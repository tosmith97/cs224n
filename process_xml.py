
import xml.etree.ElementTree
import sys
import collections
import os
import fnmatch
# adapting http://stackoverflow.com/questions/1912434/how-do-i-parse-xml-in-python (ElemlentTree)

# Constants
kWindowSize = 4

def main():
    '''
    Invocation: python process_xml.py <filename>
    '''
    #assert(len(argv) == 2)
    #filename = sys.argv[-1]
    pass

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
    root = xml.etree.ElementTree.parse(filename).getroot()
    return create_skipgram(get_word_sequence(root), kWindowSize)

def get_sense_name(element):
    suffix = element.get('sense')
    if suffix is None:
        suffix = ''
    return element.get('text') + suffix

def get_word_sequence(root):
    return [get_sense_name(word) for word in root]

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