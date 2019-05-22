import os
import csv
import shutil
import cPickle as pkl
import numpy as np

if __name__ == '__main__':
    
    #extract the news vectors for every day
    with open('imdb_100.npy','r') as f:
        train_vec_pos = np.load(f)
        print train_vec_pos.shape

    
    '''
    #compute the average vector for every day
    print "computing average vector..."
    #train_pos
    idx = 0
    train_avg_pos = []
    find = 0
    for i in train_pos:
        tmp = np.zeros(train_vec_pos[0].shape)
        for j in range(0,i):
            tmp += train_vec_pos[idx]
            idx += 1
        if i != 0:
            tmp = tmp/i
        train_avg_pos.append(tmp)

    #train_neg
    idx = 0
    train_avg_neg = []
    find = 0
    for i in train_neg:
        tmp = np.zeros(train_vec_neg[0].shape)
        for j in range(0,i):
            tmp += train_vec_neg[idx]
            idx += 1
        if i != 0:
            tmp = tmp/i
        train_avg_neg.append(tmp)

    #dev_pos
    idx = 0
    dev_avg_pos = []
    find = 0
    for i in dev_pos:
        tmp = np.zeros(dev_vec_pos[0].shape)
        for j in range(0,i):
            tmp += dev_vec_pos[idx]
            idx += 1
        if i != 0:
            tmp = tmp/i
        dev_avg_pos.append(tmp)

    #dev_neg
    idx = 0
    dev_avg_neg = []
    find = 0
    for i in dev_neg:
        tmp = np.zeros(dev_vec_neg[0].shape)
        for j in range(0,i):
            tmp += dev_vec_neg[idx]
            idx += 1
        if i != 0:
            tmp = tmp/i
        dev_avg_neg.append(tmp)

    #test_pos
    idx = 0
    test_avg_pos = []
    find = 0
    for i in test_pos:
        tmp = np.zeros(test_vec_pos[0].shape)
        for j in range(0,i):
            tmp += test_vec_pos[idx]
            idx += 1
        if i != 0:
            tmp = tmp/i
        test_avg_pos.append(tmp)

    #test_neg
    idx = 0
    test_avg_neg = []
    find = 0
    for i in test_neg:
        tmp = np.zeros(test_vec_neg[0].shape)
        for j in range(0,i):
            tmp += test_vec_neg[idx]
            idx += 1
        if i != 0:
            tmp = tmp/i
        test_avg_neg.append(tmp)

    print "finished"
    '''


