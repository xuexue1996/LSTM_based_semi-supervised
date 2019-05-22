import numpy as np
from numpy import dtype, fromstring, float32 as REAL

class WordLoader(object):
    def load_word_vector(self, fname, wordlist, dim, binary=None):
        if binary == None:
            if fname.endswith('.txt'):
                binary = False
            elif fname.endswith('.bin'):
                binary = True
            else:
                raise NotImplementedError('Cannot infer binary from %s' % (fname))

        vocab = {}
        with open(fname) as fin:
            lines = fin.readlines()
            vocab_size = len(lines)
            vec_size = 300
            if binary:
                pass
            else:
                for line_no, line in enumerate(lines):
                    try:
                        parts = line.strip().split(' ')
                        if len(parts) != vec_size + 1:
                            print "Wrong line: %s %s\n" % (line_no, line)
                        word, weights = parts[0], map(REAL, parts[1:])
                        vocab[unicode(word)] = weights
                    except:
                        pass
        print len(vocab)
        return vocab
