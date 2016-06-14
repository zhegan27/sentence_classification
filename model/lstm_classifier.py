
import numpy as np
import theano
import theano.tensor as tensor
from theano import config

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import dropout, numpy_floatX
from utils import uniform_weight, zero_bias

from lstm_layers import param_init_encoder, encoder

# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

""" init. parameters. """  

def init_params(options,W):
    
    n_h = options['n_h']
    n_y = options['n_y']
    
    params = OrderedDict()
    # W is initialized by the pretrained word embedding
    params['Wemb'] = W.astype(config.floatX)
    # otherwise, W will be initialized randomly
    # n_words = options['n_words']
    # n_x = options['n_x'] 
    # params['Wemb'] = uniform_weight(n_words,n_x)
    
    # bidirectional LSTM
    params = param_init_encoder(options,params,prefix="lstm_encoder")
    params = param_init_encoder(options,params,prefix="lstm_encoder_rev")
    
    params['Wy'] = uniform_weight(2*n_h,n_y)
    params['by'] = zero_bias(n_y)                                     

    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
        #tparams[kk].tag.test_value = params[kk]
    return tparams
    
""" Building model... """

def build_model(tparams,options):
    
    trng = RandomStreams(SEED)
    
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))
    
    # input sentence: n_steps * n_samples
    x = tensor.matrix('x', dtype='int32')
    mask = tensor.matrix('mask', dtype=config.floatX)
    
    # label: (n_samples,)
    y = tensor.vector('y',dtype='int32')

    n_steps = x.shape[0] # the length of the longest sentence in this minibatch
    n_samples = x.shape[1] # how many samples we have in this minibatch
    n_x = tparams['Wemb'].shape[1] # the dimension of the word-embedding
    
    emb = tparams['Wemb'][x.flatten()].reshape([n_steps,n_samples,n_x])  
    emb = dropout(emb, trng, use_noise)
                        
    # encoding of the sentence, size of n_samples * n_h                                                               
    h_encoder = encoder(tparams, emb, mask=mask, prefix='lstm_encoder')
    h_encoder_rev = encoder(tparams, emb[::-1], mask=mask[::-1], prefix='lstm_encoder_rev')
    
    # size of n_samples * (2*n_h) 
    z = tensor.concatenate((h_encoder,h_encoder_rev),axis=1) 
    z = dropout(z, trng, use_noise)  
    
    # this is the label prediction you made 
    # size of n_samples * n_y
    pred = tensor.nnet.softmax(tensor.dot(z, tparams['Wy'])+tparams['by'])
    
    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    # get the expression of how we calculate the cost function
    # i.e. corss-entropy loss
    index = tensor.arange(n_samples)
    cost = -tensor.log(pred[index, y] + 1e-6).mean()                          

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost
    

