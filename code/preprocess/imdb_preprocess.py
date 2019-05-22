"""
This script is what created the dataset pickled.

1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.

2) Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract it in the current directory.

3) Then run this script.
"""

dataset_path='/Users/jihanxue/Documents/atae_lstm_wen-master/aclImdb/'

import numpy
import cPickle as pkl

from collections import OrderedDict

import glob
import os
import nltk
from nltk.corpus import stopwords
from subprocess import Popen, PIPE


def dlfilter(content):
    words=content.split(' ')
    word_list=[filter(str.isalpha, word.lower()) for word in words]

    filtered_word_list =[]
    for word in word_list: # iterate over word_list
        if word not in stopwords.words('english'):
            filtered_word_list.append(word)
    content = " ".join(filtered_word_list)
    return content

def build_dict(path):
    sentences = []
    currdir = os.getcwd()
    os.chdir('%s/pos/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentence = dlfilter(f.readline().strip())
            sentence = nltk.sent_tokenize(sentence.decode('utf-8').lower())
            sentences.append(" ".join(sentence))
    os.chdir('%s/neg/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentence = dlfilter(f.readline().strip())
            sentence = nltk.sent_tokenize(sentence.decode('utf-8').lower())
            sentences.append(sentence)
    os.chdir(currdir)


    print 'Building dictionary..',
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = numpy.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict

def build_data(path1,path2):
    '''
    print("generating train pos...")
    sentences = []
    os.chdir('%s/pos/' % path1)
    idx = 0
    n = 0
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentence = dlfilter(f.readline().strip())
            sentence = nltk.sent_tokenize(sentence.decode('utf-8').lower())
            sentences.append(" ".join(sentence))
            n+=1
            print n

    #write the idx,sent,label into the file
    txtName = "/Users/jihanxue/Documents/atae_lstm_wen-master/aclImdb//Documents/atae_lstm_wen-master/imdb_train_pos.txt"
    f = open(txtName, "w")
    for i in range(len(sentences)):
        f.write(str(idx)+"\n")
        f.write(sentences[i]+"\n")
        f.write("1\n")
        idx += 1
    f.close()
    '''
    n = 12500
    print("generating train neg...")
    idx = 12500
    os.chdir('%s/neg/' % path1)
    sentences=[]
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentence = dlfilter(f.readline().strip())
            sentence = nltk.sent_tokenize(sentence.decode('utf-8').lower())
            sentences.append(" ".join(sentence))
            n += 1
            print n
    txtName = "/Users/jihanxue/Documents/atae_lstm_wen-master/imdb_train_neg.txt"
    f = open(txtName, "w")
    for i in range(len(sentences)):
        f.write(str(idx) + "\n")
        f.write(sentences[i] + "\n")
        f.write("0\n")
        idx += 1
    f.close()

    print("generating test pos...")
    sentences = []
    os.chdir('%s/pos/' % path2)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentence = dlfilter(f.readline().strip())
            sentence = nltk.sent_tokenize(sentence.decode('utf-8').lower())
            sentences.append(" ".join(sentence))
            n += 1
            print n
    # write the idx,sent,label into the file
    txtName = "/Users/jihanxue/Documents/atae_lstm_wen-master/imdb_test_pos.txt"
    f = open(txtName, "w")
    for i in range(len(sentences)):
        f.write(str(idx) + "\n")
        f.write(sentences[i] + "\n")
        f.write("1\n")
        idx += 1
    f.close()


    print("generating test neg...")
    os.chdir('%s/neg/' % path2)
    sentences = []
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentence = dlfilter(f.readline().strip())
            sentence = nltk.sent_tokenize(sentence.decode('utf-8').lower())
            sentences.append(" ".join(sentence))
            n += 1
            print n

    txtName = "/Users/jihanxue/Documents/atae_lstm_wen-master/imdb_test_neg.txt"
    f = open(txtName, "w")
    for i in range(len(sentences)):
        f.write(str(idx) + "\n")
        f.write(sentences[i] + "\n")
        f.write("0\n")
        idx += 1
    f.close()


def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    path = dataset_path
    path1 = os.path.join(path, 'train')
    path2 = os.path.join(path, 'test')
    build_data(path1,path2)
    '''
    dictionary = build_dict(os.path.join(path, 'train'))

    train_x_pos = grab_data(path+'train/pos', dictionary)
    train_x_neg = grab_data(path+'train/neg', dictionary)
    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_x_pos = grab_data(path+'test/pos', dictionary)
    test_x_neg = grab_data(path+'test/neg', dictionary)
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    f = open('imdb.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()

    f = open('imdb.dict.pkl', 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()
    '''
if __name__ == '__main__':
    main()
