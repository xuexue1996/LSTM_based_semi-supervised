# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import argparse
import time
import collections
from WordLoader import WordLoader
from DocLoader import DocLoader
import cPickle as pkl
import theano.tensor as tensor
class AttentionLstm(object):
    def __init__(self, wordlist, argv, doc_num=0):
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, default='lstm')
        parser.add_argument('--rseed', type=int, default=int(1000*time.time()) % 19491001)
        parser.add_argument('--dim_word', type=int, default=300)
        parser.add_argument('--dim_hidden', type=int, default=300)#500
        parser.add_argument('--dim_doc', type=int, default=300)#300
        parser.add_argument('--grained', type=int, default=2, choices=[2])
        parser.add_argument('--regular', type=float, default=0.001)
        parser.add_argument('--word_vector', type=str, default='/home/xue/nlp/supervised-learning/data/glove.840B.300d.txt')
        parser.add_argument('--doc_vector', type=str, default='data/imdbencode3.npy')#imdbencode
       # parser.add_argument('--reload', type=str, default=True)
        parser.add_argument('--reload', type=str, default='best_model13.pkl')
        args, _ = parser.parse_known_args(argv)        

        self.name = args.name
        self.srng = RandomStreams(seed=args.rseed)
        self.dim_word, self.dim_hidden = args.dim_word, args.dim_hidden
        self.dim_doc = args.dim_doc
        self.grained = args.grained
        self.regular = args.regular
        self.num = len(wordlist) + 1
        self.doc_num = doc_num


        if args.reload:
            print('reloading model...')
            
            self.load_param(args.reload)
            print(self.Vd.get_value())
        else:
            print('building model...')
            self.init_param()
            self.load_word_vector(args.word_vector, wordlist)
            self.load_doc_vector(args.doc_vector)
        self.init_function()
    
    def init_param(self):
        def shared_matrix(dim, name, u=0, b=0):
            matrix = self.srng.uniform(dim, low=-u, high=u, dtype=theano.config.floatX) + b
            f = theano.function([], matrix)
            return theano.shared(f(), name=name)

        u = lambda x : 1 / np.sqrt(x)

        dimc, dimh, dimd = self.dim_word, self.dim_hidden, self.dim_doc
        dim_lstm_para = dimh + dimc

        self.Vw = shared_matrix((self.num, dimc), 'Vw', 0.01)
        self.Wi = shared_matrix((dimh, dim_lstm_para), 'Wi', u(dimh))
        self.Wo = shared_matrix((dimh, dim_lstm_para), 'Wo', u(dimh))
        self.Wf = shared_matrix((dimh, dim_lstm_para), 'Wf', u(dimh))
        self.Wc = shared_matrix((dimh, dim_lstm_para), 'Wc', u(dimh))
        self.bi = shared_matrix((dimh, ), 'bi', 0.)
        self.bo = shared_matrix((dimh, ), 'bo', 0.)
        self.bf = shared_matrix((dimh, ), 'bf', 0.)
        self.bc = shared_matrix((dimh, ), 'bc', 0.)
        self.Ws = shared_matrix((dimh, self.grained), 'Ws', u(dimh))
        self.bs = shared_matrix((self.grained, ), 'bs', 0.)
        self.h0, self.c0 = np.zeros(dimh, dtype=theano.config.floatX), np.zeros(dimh, dtype=theano.config.floatX)
        self.params = [self.Vw, self.Wi, self.Wo, self.Wf, self.Wc, self.bi, self.bo, self.bf, self.bc, self.Ws, self.bs]
        
        self.Wh = shared_matrix((dimh, dimh), 'Wh', u(dimh))
        self.Wv = shared_matrix((dimd, dimd), 'Wv', u(dimh))
        self.w = shared_matrix((dimd, ), 'w', 0.)
        self.Wp = shared_matrix((dimh, dimh), 'Wp', u(dimh))
        self.Wx = shared_matrix((dimh, dimh), 'Wx', u(dimh))
        self.params.extend([self.Wh, self.Wv, self.w, self.Wp, self.Wx])
     
        self.Vd = shared_matrix((self.doc_num, dimd), 'Vd', 0.01)
        self.params.extend([self.Vd])
    
    def load_param(self, path):
        params = pkl.load(open(path,"rb")) 
        self.Vw = params['Vw']
        self.Wi = params['Wi']
        self.Wo = params['Wo']
        self.Wf = params['Wf']
        self.Wc = params['Wc']
        self.bi = params['bi']
        self.bo = params['bo']
        self.bf = params['bf']
        self.bc = params['bc']
        self.Ws = params['Ws']
        self.bs = params['bs']
        self.h0, self.c0 = params['h0'],params['c0'] 
        self.params = [self.Vw, self.Wi, self.Wo, self.Wf, self.Wc, self.bi, self.bo, self.bf, self.bc, self.Ws, self.bs]
        
        self.Wh = params['Wh']
        self.Wv = params['Wv']
        self.w = params['w']
        self.Wp = params['Wp']
        self.Wx = params['Wx']
        self.params.extend([self.Wh, self.Wv, self.w, self.Wp, self.Wx])

        self.Vd = params['Vd']
        self.params.extend([self.Vd])


    def save_param(self, path):
        params = collections.OrderedDict()
        params['Vw'] = self.Vw
        params['Wi'] = self.Wi
        params['Wo'] = self.Wo
        params['Wf'] = self.Wf
        params['Wc'] = self.Wc
        params['bi'] = self.bi
        params['bo'] = self.bo
        params['bf'] = self.bf
        params['bc'] = self.bc
        params['Ws'] = self.Ws
        params['bs'] = self.bs
        params['h0'] = self.h0
        params['c0'] = self.c0
        params['Wh'] = self.Wh
        params['Wv'] = self.Wv
        params['w'] = self.w
        params['Wp'] = self.Wp
        params['Wx'] = self.Wx
        params['Vd'] = self.Vd
        pkl.dump(params,open(path, 'wb'), -1)
    def init_function(self):
        self.seq_idx = T.lvector() 
        self.tar_scalar = T.lscalar()
        self.solution = T.matrix()
        self.seq_matrix = T.take(self.Vw, self.seq_idx, axis=0)
        self.tar_vector = T.take(self.Vd, self.tar_scalar, axis=0)

        h, c = T.zeros_like(self.bf, dtype=theano.config.floatX), T.zeros_like(self.bc, dtype=theano.config.floatX)

        def encode(x_t, h_fore, c_fore, tar_vec):
            v = T.concatenate([h_fore, x_t])
            f_t = T.nnet.sigmoid(T.dot(self.Wf, v) + self.bf)
            i_t = T.nnet.sigmoid(T.dot(self.Wi, v) + self.bi)
            o_t = T.nnet.sigmoid(T.dot(self.Wo, v) + self.bo)
            c_next = f_t * c_fore + i_t * T.tanh(T.dot(self.Wc, v) + self.bc)
            h_next = o_t * T.tanh(c_next)
            return h_next, c_next

        scan_result, _ = theano.scan(fn=encode, sequences=[self.seq_matrix], outputs_info=[h, c], non_sequences=[self.tar_vector])
        embedding = scan_result[0] # embedding in there is a matrix, include[h_1, ..., h_n]

        # attention
        
        tmp1 = tensor.sum(self.tar_vector**2, axis=0).reshape((1, -1))
        tmp2 = tensor.sum(embedding**2, axis=1).reshape((1,-1))
        tmp3 = tensor.dot(tmp1, tmp2)

        e = tensor.dot(embedding, self.tar_vector) / tensor.sqrt(tmp3)
        alpha_tmp = T.nnet.softmax(e)
        '''
        matrix_doc = T.zeros_like(embedding, dtype=theano.config.floatX)[:,:self.dim_doc] + self.tar_vector
        hhhh = T.concatenate([T.dot(embedding, self.Wh), T.dot(matrix_doc, self.Wv)], axis=1)
        M_tmp = T.tanh(hhhh)
        alpha_tmp = T.nnet.softmax(T.dot(M_tmp, self.w))
        '''
        r = T.dot(alpha_tmp, embedding)
        h_star = T.tanh(T.dot(r, self.Wp) + T.dot(embedding[-1], self.Wx))
        embedding = h_star # embedding in there is a vector, represent h_n_star
 
        # dropout
        embedding_for_train = embedding * self.srng.binomial(embedding.shape, p = 0.5, n = 1, dtype=embedding.dtype)
        embedding_for_test = embedding * 0.5
            
        self.pred_for_train = T.nnet.softmax(T.dot(embedding_for_train, self.Ws) + self.bs)
        self.pred_for_test = T.nnet.softmax(T.dot(embedding_for_test, self.Ws) + self.bs)

        self.l2 = sum([T.sum(param**2) for param in self.params]) - T.sum(self.Vw**2)
        self.loss_sen = -T.tensordot(self.solution, T.log(self.pred_for_train), axes=2)
        self.loss_l2 = 0.7 * self.l2 * self.regular
        self.loss = self.loss_sen + self.loss_l2

        grads = T.grad(self.loss, self.params)
        self.updates = collections.OrderedDict()
        self.grad = {}
        for param, grad in zip(self.params, grads):
            g = theano.shared(np.asarray(np.zeros_like(param.get_value()), \
                    dtype=theano.config.floatX))
            self.grad[param] = g
            self.updates[g] = g + grad

        self.func_train = theano.function(
                inputs = [self.seq_idx, self.tar_scalar, self.solution, theano.In(h, value=self.h0), theano.In(c, value=self.c0)],
                outputs = [self.loss, self.loss_sen, self.loss_l2],
                updates = self.updates,
                on_unused_input='warn')

        self.func_test = theano.function(
                inputs = [self.seq_idx, self.tar_scalar, theano.In(h, value=self.h0), theano.In(c, value=self.c0)],
                outputs = self.pred_for_test,
                on_unused_input='warn')
    
    def load_word_vector(self, fname, wordlist):
        Vw = pkl.load(open('Vw_mr.pkl','rb'))

#        loader = WordLoader()
#        dic = loader.load_word_vector(fname, wordlist, self.dim_word)
#        not_found = 0
#        Vw = self.Vw.get_value()
#        for word, index in wordlist.items():
#            try:    
#                Vw[index] = dic[word]
#                
#            except:
#                not_found += 1
#        print "not found:", not_found
#        print 'saving wv'
#        pkl.dump(Vw,open('Vw_mr.pkl','wb'),-1) 
#
#        print 'Vw ',len(Vw),len(Vw[0])
#
        self.Vw.set_value(Vw)

    def load_doc_vector(self, fname):
        loader = DocLoader()
        dic = loader.load_doc_vector(fname, self.dim_doc)

        not_found = 0
        Vd = self.Vd.get_value()
        for index in range(Vd.shape[0]):
            try:
                Vd[index] = dic[index]
            except:
                not_found += 1
        self.Vd.set_value(Vd)

