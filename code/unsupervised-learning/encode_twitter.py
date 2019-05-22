from __future__ import print_function
from six.moves import xrange
import six.moves.cPickle as pickle
from nltk.tokenize import word_tokenize as tokenize

import gzip
import os

import numpy
import theano

import cPickle

loadnewdata = False 
file_name = "Data/chageimdbluanma.txt" 
def prepare_data(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
   # print(lengths)
    maxlen = numpy.max(lengths)

    x_source = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_target = numpy.zeros((maxlen, n_samples)).astype('int64')
    mask_source = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    mask_target = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x_source[maxlen-lengths[idx]:, idx] = s
        mask_source[maxlen-lengths[idx]:, idx] = 1.
        x_target[:lengths[idx], idx] = s
        mask_target[:lengths[idx], idx] = 1.

    return x_source, x_target, mask_source, mask_target, x_target


def load_data(path=file_name, n_words=50000, valid_portion=0.1, maxlen=100, minlen=5, total_samples=1000000,
              sort_by_len=False):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (0).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############

    # Load the dataset 
    if loadnewdata:
        path = 'Data/' + path

        if path.endswith(".gz"):
            f = gzip.open(path, 'r')
        else:
            f = open(path, 'r')    
    
        fout = open(file_name, 'w')
        wdict = {}
        n = 0
        for line in f:
            lines = line.strip().split(' ')
            if len(lines) >= minlen and len(lines) <= maxlen:
                n += 1
                fout.write(' '.join(lines) + '\n')
                #print (' '.join(lines))
            
                for w in lines:
                    if wdict.has_key(w):
                        t = wdict[w]
                    else:
                        t = 0
                    t += 1
                    wdict[w] = t
        
          #  if n > total_samples:
               # break
        fout.close()
    
        wdic_sorted = sorted(wdict.iteritems(), key = lambda kv:kv[1], reverse = True)
        for i in xrange(len(wdic_sorted)):
            wdict[wdic_sorted[i][0]] = i + 1
        
        fdic = open('word_index.pkl', 'w')
        cPickle.dump(wdict, fdic)
        fdic.close()

        f.close()
    else:
        fdic = open('word_index.pkl', 'r')
        wdict = cPickle.load(fdic)
        fdic.close()
        
    train_set_x = []
    train_set_y = []
    fin = open(file_name, 'r')
    notfound = 0
    for line in fin:
        lines = line.strip().split(' ')
        sen = []
        for word in lines:
            if(wdict.has_key(word)):
                sen.append(wdict[word])
            else:
                sen.append(0)
                notfound += 1
        train_set_x.append(sen)
        train_set_y.append(sen)
    print(notfound)
    fin.close()
    train_set = (train_set_x, train_set_y)
    
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set_x, train_set_y):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y
        
    train_set_x, train_set_y = train_set
    
    # split training set into validation set
    '''
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    '''
    train_set = (train_set_x, train_set_y)
    valid_set = (train_set_x, train_set_y)

    def remove_unk(x):
        return [[0 if w >= n_words else w for w in sen] for sen in x]
    
    def add_end(x):
        for sen in x:
            sen.append(n_words)
        return x

    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    
    train_set_x = add_end(train_set_x)
    valid_set_x = add_end(valid_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)

    return train, valid

if __name__ == '__main__':
    load_data()
