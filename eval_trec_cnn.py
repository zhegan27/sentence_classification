'''
Build a CNN Classifier Trained on TREC dataset
Zhe Gan @zg27@duke.edu, June, 13, 2016
Duke University, ECE
'''

import time
import logging
import cPickle

import numpy as np
import theano
import theano.tensor as tensor

from model.cnn_classifier import init_params, init_tparams
from model.cnn_classifier import build_model

from model.optimizers import Adam
from model.utils import get_minibatches_idx
from model.utils import numpy_floatX, unzip, zipp

""" Training the model. """

""" used to calculate the prediction error. """

def pred_error(f_pred, prepare_data, data, iterator, max_len, n_words, filter_h):
    
    """ compute the prediction error. 
    """
    valid_err = 0
    for _, valid_index in iterator:
                                  
        x = [data[0][t] for t in valid_index]
        x = prepare_data(x,max_len,n_words,filter_h)
        preds = f_pred(x)
        targets = np.array([data[1][t] for t in valid_index],dtype='int32')
        valid_err += (preds == targets).sum()
                
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err

""" used to preprocess the dataset. """
 
def prepare_data(seqs_x, max_len, n_words, filter_h):
    
    pad = filter_h -1
    x = []   
    for rev in seqs_x:    
        xx = []
        for i in xrange(pad):
            # we need pad the special <pad_zero> token.
            xx.append(n_words-1)
        for idx in rev:
            xx.append(idx)
        while len(xx) < max_len + 2*pad:
            # we need pad the special <pad_zero> token.
            xx.append(n_words-1)
        x.append(xx)
    x = np.array(x,dtype='int32')
    return x

def train_classifier(train, valid, test, W, n_words=10000, img_w=300, max_len=40, 
    feature_maps=100, filter_hs=[3,4,5], dropout_val=0.5, patience=10, 
    max_epochs=20, lrate=0.0002, batch_size=50, valid_batch_size=50, dispFreq=10, 
    validFreq=100, saveFreq=200, saveto = 'trec_cnn_result.npz'):
        
    """ train, valid, test : datasets
        W : the word embedding initialization
        n_words : vocabulary size
        img_w : word embedding dimension, must be 300.
        max_len : the maximum length of a sentence 
        feature_maps : the number of feature maps we used 
        filter_hs: the filter window sizes we used
	dropout_val: dropput probability
        patience : Number of epoch to wait before early stop if no progress
        max_epochs : The maximum number of epoch to run
        lrate : learning rate
        batch_size : batch size during training
        valid_batch_size : The batch size used for validation/test set
        dispFreq : Display to stdout the training progress every N updates
        validFreq : Compute the validation error after this number of update.
        saveFreq: save the result after this number of update.
        saveto: where to save the result.
    """

    img_h = max_len + 2*(filter_hs[-1]-1)
    
    options = {}
    options['n_words'] = n_words
    options['img_w'] = img_w
    options['img_h'] = img_h
    options['feature_maps'] = feature_maps
    options['filter_hs'] = filter_hs
    options['patience'] = patience
    options['max_epochs'] = max_epochs
    options['lrate'] = lrate
    options['batch_size'] = batch_size
    options['valid_batch_size'] = valid_batch_size
    options['dispFreq'] = dispFreq
    options['validFreq'] = validFreq
    
    logger.info('Model options {}'.format(options))
    
    logger.info('{} train examples'.format(len(train[0])))
    logger.info('{} valid examples'.format(len(valid[0])))
    logger.info('{} test examples'.format(len(test[0])))

    logger.info('Building model...')
    
    n_y = np.max(train[1]) + 1
    options['n_y'] = n_y
    
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    """ 

    filter_w = img_w
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
        
    options['filter_shapes'] = filter_shapes
    options['pool_sizes'] = pool_sizes
    
    params = init_params(options,W)
    tparams = init_tparams(params)

    (use_noise, x, y, f_pred_prob, f_pred, cost) = build_model(tparams,options)
    
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = Adam(tparams, cost, [x, y], lr)

    logger.info('Training model...')
    
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    estop = False  # early stop
    history_errs = []
    best_p = None
    bad_counter = 0    
    uidx = 0  # the number of update done
    start_time = time.time()
    
    zero_vec_tensor = tensor.vector()
    zero_vec = np.zeros(img_w).astype(theano.config.floatX)
    set_zero = theano.function([zero_vec_tensor], updates=[(tparams['Wemb'], tensor.set_subtensor(tparams['Wemb'][n_words-1,:], zero_vec_tensor))])
    
    try:
        for eidx in xrange(max_epochs):

            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(dropout_val)

                y = np.array([train[1][t] for t in train_index]).astype('int32')
                x = [train[0][t]for t in train_index]
                x = prepare_data(x,max_len,n_words,filter_hs[-1])

                cost = f_grad_shared(x, y)
                f_update(lrate)
                # the special token does not need to update.
                set_zero(zero_vec)

                if np.isnan(cost) or np.isinf(cost):
                    logger.info('NaN detected')
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    logger.info('Epoch {} Update {} Cost {}'.format(eidx, uidx, cost))
                    
                if np.mod(uidx, saveFreq) == 0:
                    logger.info('Saving ...')
                    
                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                        np.savez(saveto, history_errs=history_errs, **params)
                    
                    logger.info('Done ...')

                if np.mod(uidx, validFreq) == 0:
                    
                    use_noise.set_value(0.)
                    
                    train_err = pred_error(f_pred, prepare_data, train, kf, max_len,n_words, filter_hs[-1])
                    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid, max_len,n_words, filter_hs[-1])
                    test_err = pred_error(f_pred, prepare_data, test, kf_test, max_len,n_words, filter_hs[-1])
                    history_errs.append([valid_err, test_err, train_err])
                   
                    if (uidx == 0 or
                        valid_err <= np.array(history_errs)[:,0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    logger.info('Train {} Valid {} Test {}'.format(train_err, valid_err, test_err))

                    if (len(history_errs) > patience and
                        valid_err >= np.array(history_errs)[:-patience,0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            
                            logger.info('Early Stop!')
                            estop = True
                            break

            if estop:
                break

    except KeyboardInterrupt:
        logger.info('Training interupted')

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)
    
    use_noise.set_value(0.)
    
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted, max_len,n_words, filter_hs[-1])
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid, max_len,n_words, filter_hs[-1])
    test_err = pred_error(f_pred, prepare_data, test, kf_test, max_len,n_words, filter_hs[-1])

    logger.info('Train {} Valid {} Test {}'.format(train_err, valid_err, test_err))
    
    np.savez(saveto, train_err=train_err,
             valid_err=valid_err, test_err=test_err,
             history_errs=history_errs, **best_p)
    
    logger.info('The code run for {} epochs, with {} sec/epochs'.format(eidx + 1, 
                 (end_time - start_time) / (1. * (eidx + 1))))
    
    return train_err, valid_err, test_err

def create_valid(train_set,valid_portion=0.10):
    
    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)

    return train, valid

if __name__ == '__main__':
    
    # https://docs.python.org/2/howto/logging-cookbook.html
    logger = logging.getLogger('eval_trec_cnn')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('eval_trec_cnn.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    
    logger.info('loading data...')
    x = cPickle.load(open("./data/trec.p","rb"))
    train, test, W, ixtoword, wordtoix= x[0], x[1], x[2], x[3], x[4]
    del x
    
    n_words = W.shape[0]
    
    length = []
    for sent in train[0]:
        length.append(len(sent))
    
    for sent in test[0]:
        length.append(len(sent))
    
    max_len = np.max(length) + 1
    
    n_words = len(ixtoword)
    # add another special token called <pad_zero>
    ixtoword[n_words] = '<pad_zero>'
    wordtoix['<pad_zero>'] = n_words
    n_words = n_words + 1
    Wemb = np.zeros((n_words,300))
    Wemb[:n_words-1] = W
    del W
    
    results = []
    # run the cnn classifier ten times
    r = range(0,10)
    for i in r:
        train0, valid = create_valid(train, valid_portion=0.10)
        [train_err, valid_err, test_err] = train_classifier(train0, valid, test, 
            Wemb, n_words=n_words, max_len=max_len)
        logger.info('try: {} test err: {}'.format(i, test_err))
        results.append(test_err)
    logger.info('final test err: {} std: {}'.format(np.mean(results),np.std(results)))   
    
