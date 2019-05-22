# -*- coding: utf-8 -*-
import numpy as np
import theano
import argparse
import time
import sys
import json
import random
from Optimizer import OptimizerList
from Evaluator import Evaluators
from DataManager import DataManager
import cPickle as pkl

from lstm_attention_attention import AttentionLstm as Model
from lstm import Lstm as Model2

#theano.config.optimizer="fast_compile"
def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = range(n)

    if shuffle:
        random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def train(model, train_data, optimizer, epoch_num, batch_size, batch_n):
    st_time = time.time()
    loss_sum = np.array([0.0, 0.0, 0.0])
    total_nodes = 0
    kf = get_minibatches_idx(len(train_data), batch_size, shuffle=True)
    i=0
    for _, train_index in kf:
        i+=1
        if i%10 == 0:
            print("batch:%d" % i)
        
        if i % 10 == 0:
            break
        
        train_data_tmp = [train_data[t] for t in train_index]
        #start = batch * batch_size
        #end = min((batch + 1) * batch_size, len(train_data))
        batch_loss, batch_total_nodes = do_train(model, train_data_tmp, optimizer)
        loss_sum += batch_loss
        total_nodes += batch_total_nodes

    return loss_sum[0], loss_sum[2]


def do_train(model, train_data, optimizer):
    eps0 = 1e-8
    batch_loss = np.array([0.0, 0.0, 0.0])
    total_nodes = 0
    for _, grad in model.grad.iteritems():
        grad.set_value(np.asarray(np.zeros_like(grad.get_value()), \
                                  dtype=theano.config.floatX))
    for item in train_data:
        sequences, target, tar_scalar, solution = item['seqs'], item['target'], item['target_index'], item['solution']
        batch_loss += np.array(model.func_train(sequences, tar_scalar, solution))
        total_nodes += len(solution)
    for _, grad in model.grad.iteritems():
        grad.set_value(grad.get_value() / float(len(train_data)))
    optimizer.iterate(model.grad)
    return batch_loss, total_nodes


def test(model, test_data, grained):
    evaluator = Evaluators[grained]()
    keys = evaluator.keys()

    def cross(solution, pred):
        #print(pred)
   #     print(solution)
        return -np.tensordot(solution, np.log(pred), axes=([0, 1], [0, 1]))
    fout = open("testlog.txt","w")
    loss = .0
    total_nodes = 0
    correct = {key: np.array([0]) for key in keys}
    wrong = {key: np.array([0]) for key in keys}
    for i,item in enumerate(test_data):
        sequences, target, tar_scalar, solution = item['seqs'], item['target'], item['target_index'], item['solution']
        pred = model.func_test(sequences, tar_scalar)
        loss += cross(solution, pred)
        total_nodes += len(solution)
        result = evaluator.accumulate(solution[-1:], pred[-1:])
        #print(result)
       # print(result['binary'])
        if result['binary'] == 0:
            fout.write(str(i)+" ")
            fout.write(str(pred))
            fout.write("\n")
            fout.flush()
    acc = evaluator.statistic()
    return loss / total_nodes, acc


if __name__ == '__main__':
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='lstm_attention17')
    parser.add_argument('--seed', type=int, default=int(1000 * time.time()))
    parser.add_argument('--dim_hidden', type=int, default=300)
    parser.add_argument('--dim_gram', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--fast', type=int, choices=[0, 1], default=0)
    parser.add_argument('--screen', type=int, choices=[0, 1], default=0)
    parser.add_argument('--optimizer', type=str, default='ADAGRAD')
    parser.add_argument('--grained', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_word_vector', type=float, default=0.000007)
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--doc_num', type=int, default=50000)
    #parser.add_argument('--reload', type=str, default=True)
    parser.add_argument('--saveto', type=str, default='best_model17.pkl')
    parser.add_argument('--reload_dic', type=str, default=False)
    #parser.add_argument('--reload_dic', type=str, default='dic.pkl')
    args, _ = parser.parse_known_args(argv)
    random.seed(args.seed)
    data = DataManager(args.dataset)
    if args.reload_dic:
        print('reloading dictionary...')
        wordlist = data.load_word(args.reload_dic)
        
    else:
        print('building dictionary...')
        wordlist = data.gen_word()
        print('saving dictionary...')
        pkl.dump(wordlist,open('dic.pkl', 'wb'), -1)
    print('%d unique words in total'%len(wordlist))
    train_data, test_data = data.gen_data(args.grained)
    random.shuffle(train_data)
    num = int(len(train_data)*0.11)
    dev_data = train_data[:num]
    train_data_new = train_data[num:]
    train_data_new = train_data
    model = Model(wordlist, argv, args.doc_num)
    batch_n = (len(train_data_new) - 1) / args.batch + 1
    optimizer = OptimizerList[args.optimizer](model.params, args.lr, args.lr_word_vector)
    details = {'loss': [], 'loss_train': [], 'loss_dev': [], 'loss_test': [], \
               'acc_train': [], 'acc_dev': [], 'acc_test': [], 'loss_l2': []}
    #print('saving params...')
    #model.save_param(args.saveto)
    print('start training...')

    print('testing...')
    now = {}
    best_acc = 0.0
    my_best_acc = 0.0
  #  now['loss_test'], now['acc_test'] = test(model, test_data, args.grained)
   # print('acc_test', now['acc_test'],'loss_test', now['loss_test'])
    for e in range(args.epoch):
        random.shuffle(train_data_new)
        now = {}
        print('epoch', e)
        now['loss'], now['loss_l2'] = train(model, train_data_new, optimizer, e, args.batch, batch_n)
        print('testing  data...')
        #now['loss_train'], now['acc_train'] = test(model, train_data_new, args.grained)
        #print('testing dev data...')
      #  random.shuffle(test_data)
      #  num = len(train_data)/10
        #now['loss_dev'], now['acc_dev'] = test(model, test_data, args.grained)
        print('loss_train', now['loss'])
        now['loss_dev'], now['acc_dev'] = test(model, dev_data, args.grained)
        print('acc_dev', now['acc_dev'],'loss_dev', now['loss_dev'])

        if best_acc < now['acc_dev']:
            best_acc = now['acc_dev']
            
      #  test_data = test_data[12500:]
            now['loss_test'], now['acc_test'] = test(model, test_data, args.grained)
            if my_best_acc < now['acc_test']:
          #  print('New best acc found, saving params...')
                  model.save_param(args.saveto)
                  my_best_acc = now['acc_test']


                  print('acc_test', now['acc_test'],'loss_test', now['loss_test'])


        for key, value in now.items():
            details[key].append(value)
        with open('result/%s.txt' % args.name, 'w') as f:
            f.writelines(json.dumps(details))
