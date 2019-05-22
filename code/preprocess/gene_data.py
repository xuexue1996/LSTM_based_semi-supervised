import xml.etree.ElementTree as ET
import sys
import os
import cPickle as pkl
import nltk
from nltk.corpus import stopwords
import numpy as np
import itertools


DUMP_FILE='nivea_list.pkl'
DATASET='dataset.pkl'
vocabulary_size=50000
unknown_token='unknown_token'

class nivea:
    def __init__(self, domain, polarity, summary, text):
        self.domain = domain
        if polarity=='negative':
            self.polarity = 0
        elif polarity=='positive':
            self.polarity=1
        else:
            self.polarity=-1
        self.summary = summary
        self.text = text

def dlfilter(content):
    words=content.split(' ')
    word_list=[filter(str.isalpha, word.lower()) for word in words]

    filtered_word_list =[]
    for word in word_list: # iterate over word_list
        if word not in stopwords.words('english'):
            filtered_word_list.append(word)
    content = " ".join(filtered_word_list)
    return content


def dump(content):
    with open(DUMP_FILE,'w') as f:
        pkl.dump(content,f)


def load():
    """
    :rtype: nivea
    """
    with open(DUMP_FILE,'r') as f:
        content=pkl.load(f)
    return content

def parse(root_path):
    fpaths=os.listdir(root_path)
    nivea_list = []
    for fp in fpaths:
        filepath=os.path.join(root_path,fp)
        try:
            root = ET.parse(filepath)
        except:
            continue
        root=root.getroot()

        for sent in root:
           nivea_list.append(nivea(sent[0].text, sent[1].text, dlfilter(sent[2].text.strip()), dlfilter(sent[3].text.strip())))


    return nivea_list

if __name__=="__main__":

    if os.path.isfile(DUMP_FILE):
        nivea_list=load()
    else:
        root_path='data/Video_Games'
        nivea_list=parse(root_path)
        dump(nivea_list)
    sentences=[]
    labels=[]
    for item in nivea_list:
        labels.append(item.polarity)
        sents=nltk.sent_tokenize(item.text.decode('utf-8').lower())
        sentences.append(" ".join(sents))


    tokenized_sentences=[nltk.word_tokenize(sent) for sent in sentences]
    print 'sentences:',len(tokenized_sentences),'label:',len(labels)

    word_freq=nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens" % len(word_freq.items())

    vocab=word_freq.most_common(vocabulary_size-1)
    index2word=[x[0] for x in vocab]
    index2word.append(unknown_token)
    word2index=dict([(w,i) for i,w in enumerate(index2word)])

    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in out vocabulary is '%s' and appeared %d times." \
                %(vocab[-1][0],vocab[-1][1])


    # Replace all words not in our vocabulary with the unknown token
    fw=open('corpus20.txt','w')
    for i,sent in enumerate(tokenized_sentences):
        tokenized_sentences[i]=[w if w in word2index else unknown_token for w in sent]
        fw.write(" ".join(tokenized_sentences[i]))
        fw.write('\n')
    fw.close()

    print "\nExample sentence: '%s'" % sentences[0]
    print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
    print len(tokenized_sentences)
    maxlen=200
    pad_sentences=[]
    '''for sent in tokenized_sentences:
        min_length=min(len(sent),maxlen)
        temp=np.zeros(maxlen)
        for i in range(min_length):
            temp[i]=word2index(sent[i])
        pad_sentences.append(temp)


    # Create the training data
    #X_train=np.asarray([[word2index[w] for w in sent]for sent in tokenized_sentences])
    X_train=np.asarray(pad_sentences)
    y_train=np.asarray(labels)

    # Print an training data example
    x_example,y_example=X_train[17],y_train[17]
    #print "x:\n%s\n%s" %(" ".join([index2word[x] for x in x_example]),x_example)
    print "\ny:\n%d" % y_example

    with open(DATASET,'w') as f:
        pkl.dump((X_train,y_train),f)'''

