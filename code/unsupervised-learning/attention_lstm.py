'''
Build a tweet sentiment analyzer
'''

from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
theano.allow_input_downcast=True
import encode_twitter as twitter
#import imdb
theano.config.allow_growth = False
theano.config.floatX = "float32"
#theano.config.optimizer='fast_compile'
datasets = {'twitter': (twitter.load_data, twitter.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

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


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
   # randn = numpy.random.rand(options['n_words']+1,
            #                  options['dim_proj'])
    randn = numpy.load('embedding.npy')
    params['Wemb'] = ( randn).astype(config.floatX)
    # attention
    params['Ch'] = (0.01 * numpy.random.uniform(-1.0,
                                               1.0,
                                               (options['dim_proj'],options['dim_proj'])).astype(config.floatX))
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    params = get_layer(options['decoder'])[0](options,
                                              params,
                                              prefix=options['decoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, state_pre=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None
    
    dim_proj = options['dim_proj']
    if state_pre is None:
        state_pre = [tensor.alloc(numpy_floatX(0.),
                                  n_samples,
                                  dim_proj),
                     tensor.alloc(numpy_floatX(0.),
                                  n_samples,
                                  dim_proj)]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=state_pre,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0], rval[1]


def attention_lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, state_pre=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    dim_proj = options['dim_proj']
    if state_pre is None:
        state_pre = [tensor.alloc(numpy_floatX(0.),
                                  n_samples,
                                  dim_proj),
                     tensor.alloc(numpy_floatX(0.),
                                  n_samples,
                                  dim_proj)]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_, c_lstm):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        h_i = tensor.mean(h_,axis=0)
        c_lstm = tensor.mean(c_lstm,axis=1)
        tmp1 = tensor.sum(c_lstm**2, axis=1).reshape((-1, 1))
        tmp2 = tensor.sum((h_i.T)**2, axis=0).reshape((1,-1))
        tmp3 = tensor.dot(tmp2, tmp1.T)
        e = tensor.dot(c_lstm,h_i.T) / tensor.sqrt(tmp3)
        a = tensor.nnet.softmax(e)
        c_i = tensor.dot(a, c_lstm)
        tmp4 = tensor.dot(c_i, tparams['Ch'])
        #preact += [tmp4,tmp4,tmp4,tmp4]
        tmp4 = tensor.concatenate([tmp4,tmp4,tmp4,tmp4],axis=1)
        preact += tmp4
        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])
    h_pre, c_lstm = state_pre
    state_pre = [h_pre[-1], c_lstm[-1]]
    #state_pre = [state_pre[0], state_pre[1][-1]]
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=state_pre,
                                non_sequences=c_lstm,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)

    return rval[0], rval[1]

# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm_encoder': (param_init_lstm, lstm_layer), 'lstm_decoder':(param_init_lstm, attention_lstm_layer)}


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x_source, x_target, mask_source, mask_target, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x_source, x_target, mask_source, mask_target, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_encoder(tparams, options):
    x = tensor.matrix('x_source', dtype='int64')
    mask = tensor.matrix('mask_source', dtype=config.floatX)

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj, state_hid = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    
    encode = proj, state_hid
    #encode = proj[-1], state_hid[-1]
    #f_encode = theano.function([x, mask], encode, name='f_encode')

    return x, mask, encode

def build_decoder(tparams, options, encode):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x_target', dtype='int64')
    mask = tensor.matrix('mask_target', dtype=config.floatX)
    y = tensor.matrix('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    
    proj, state_hid = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix=options['decoder'],
                                            mask=mask, state_pre=encode)
    proj = tensor.concatenate([tensor.shape_padleft(encode[0][-1]), proj[:proj.shape[0]-1]])
    if options['decoder'] == 'lstm':
        proj = proj * mask[:, :, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)
    proj = tensor.reshape(proj, [(n_timesteps) * n_samples, proj.shape[2]])

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
    
    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6
    
    cost = -tensor.log(pred[tensor.arange(n_timesteps * n_samples), y.flatten()] + off) * mask.flatten()
    cost = cost.sum() / mask.sum()
    
    pred = tensor.reshape(pred, [n_timesteps, n_samples, pred.shape[1]])
    pred = pred * mask[:, :, None]

    #f_decode_prob = theano.function([x, mask], pred, name='f_decode_prob')
    #f_decode = theano.function([x, mask], pred.argmax(axis=2), name='f_decode')

    return use_noise, x, mask, y, pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    total_words = 0
    for _, valid_index in iterator:
        x_source, x_target, mask_source, mask_target, y = prepare_data([data[0][t] for t in valid_index],
                                                                       numpy.array(data[1])[valid_index],
                                                                       maxlen=None)
        preds = f_pred(x_source, x_target, mask_source, mask_target)
        targets = y
        valid_err += ((preds == targets) * mask_target).sum()
        total_words += mask_target.sum()
    valid_err = 1. - numpy_floatX(valid_err) / total_words

    return valid_err


def train_lstm(
    dim_proj=300,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=10,  # The maximum number of epoch to run
    dispFreq=1,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words=50000,  # Vocabulary size
    total_samples=1000000,
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm_encoder',  
    decoder='lstm_decoder',  
    saveto='allamzazon.npz',  # The best model will be saved there  amazon_128_50.npz
    validFreq=370,  # Compute the validation error after this number of update.
    saveFreq=-1,  # Save the parameters after every saveFreq updates
    minlen=-1,
    maxlen=10000,  # Sequence longer then this get ignored
    batch_size=4,  # The batch size during training.512 64
    valid_batch_size=8,  # The batch size used for validation/test set.64  16
    dataset='twitter',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=False,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=True,  # Path to a saved model we want to start from.
):

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    load_data, prepare_data = get_dataset(dataset)

    print('Loading data')
    train, valid = load_data(n_words=n_words, valid_portion=0.1,
                                   minlen=minlen, maxlen=maxlen, total_samples=total_samples)

    ydim = n_words + 2

    model_options['ydim'] = ydim

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('changeencodeluanma0__0.npz', params)#earlystopmodel9__11184

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (x_source, mask_source, encode) = build_encoder(tparams, model_options)
    (use_noise, x_target, mask_target, y, pred, cost) = build_decoder(tparams, model_options, encode)
    
    f_decode = theano.function([x_source, x_target, mask_source, mask_target], pred.argmax(axis=2), name='f_decode')

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x_source, x_target, mask_source, mask_target, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x_source, x_target, mask_source, mask_target, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x_source, x_target, 
                                        mask_source, mask_target, 
                                        y, cost)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    batchnum = len(train[0]) // batch_size
    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size
    fout  = open("luanmalog2.txt","a")
    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    average = 0
    
    try:
        for eidx in range(max_epochs):
            average = 0
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
            temp = 0
            for _, train_index in kf:
              #  break
                uidx += 1
                temp += 1
               # if(eidx == 0 and temp <= 22300):continue
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x_source, x_target, mask_source, mask_target, y = prepare_data(x, y)
                n_samples += x_source.shape[1]

                cost = f_grad_shared(x_source, x_target, mask_source, mask_target, y)
               # f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    nowtime = time.time()
                    print('Epoch ', eidx, 'Update ', uidx - eidx * batchnum, '/', batchnum, 'Cost ', cost, 
                          'Time cost ', nowtime - start_time, 'Expected epoch time cost ', (nowtime-start_time) * batchnum / uidx)
                    fout.write('Epoch '+str(eidx) + 'Update '+str(uidx - eidx * batchnum)+'/'+str( batchnum) + 'Cost '+str(cost))
                    average += cost
                    fout.write("\n")
                    fout.flush()
                   # break;

                    
                #    pickle.dump(model_options, open('savemodel/cnoupdateglove%s%d__%d.pkl'% (saveto,eidx,temp), 'wb'), -1)
                    
                    print('Done')
                '''
                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    #train_err = pred_error(f_decode, prepare_data, train, kf)
                    train_err = 0
                    valid_err = pred_error(f_decode, prepare_data, valid, kf_valid)

                    history_errs.append(valid_err)

                    if (best_p is None or
                        valid_err <= numpy.array(history_errs).min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print( ('Train ', train_err, 'Valid ', valid_err,) )

                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break
                '''
            
            print('Saving...')
            temp = 0
            if best_p is not None:
                params = best_p
            else:
                params = unzip(tparams)
            tsaveto = 'savemodel/changeencodeluanma%d__%d'% (eidx,temp)
            numpy.savez(tsaveto, history_errs=history_errs, **params)
            print('Seen %d samples' % n_samples)
            average = average / 12500
            fout.write("averge\n")
            fout.write(str(average))
            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")
    fout.close()
    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_decode, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_decode, prepare_data, valid, kf_valid)

    print( 'Train ', train_err, 'Valid ', valid_err, )
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, 
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print('Training took %.1fs' % (end_time - start_time))
    return train_err, valid_err


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_lstm()
