
import numpy as np
import theano
import theano.tensor as tensor
from theano import config

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import dropout, numpy_floatX
from utils import uniform_weight, zero_bias
from utils import _p

from cnn_layers import param_init_encoder, encoder

# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

""" init. parameters. """  

def init_params(options,W):
    
    params = OrderedDict()
    # W is initialized by the pretrained word embedding
    params['Wemb'] = W.astype(config.floatX)
    # otherwise, W will be initialized randomly
    # n_words = options['n_words']
    # n_x = options['n_x'] 
    # params['Wemb'] = uniform_weight(n_words,n_x)
    
    length = len(options['filter_shapes'])
    for idx in range(length):
        params = param_init_encoder(options['filter_shapes'][idx],params,prefix=_p('cnn_encoder',idx))
    
    n_h = options['feature_maps'] * length
    params['Wy'] = uniform_weight(n_h,options['n_y'])
    params['by'] = zero_bias(options['n_y'])                                     

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
    
    # input sentence: n_samples * n_steps 
    x = tensor.matrix('x', dtype='int32')
    # label: (n_samples,)
    y = tensor.vector('y',dtype='int32')
    
    layer0_input = tparams['Wemb'][tensor.cast(x.flatten(),dtype='int32')].reshape((x.shape[0],1,x.shape[1],tparams['Wemb'].shape[1])) 
    layer0_input = dropout(layer0_input, trng, use_noise)
 
    layer1_inputs = []
    for i in xrange(len(options['filter_hs'])):
        filter_shape = options['filter_shapes'][i]
        pool_size = options['pool_sizes'][i]
        conv_layer = encoder(tparams, layer0_input,filter_shape=filter_shape, pool_size=pool_size,prefix=_p('cnn_encoder',i))                          
        layer1_input = conv_layer
        layer1_inputs.append(layer1_input)
    layer1_input = tensor.concatenate(layer1_inputs,1)
    layer1_input = dropout(layer1_input, trng, use_noise) 
    
    # this is the label prediction you made 
    pred = tensor.nnet.softmax(tensor.dot(layer1_input, tparams['Wy']) + tparams['by'])
    
    f_pred_prob = theano.function([x], pred, name='f_pred_prob')
    f_pred = theano.function([x], pred.argmax(axis=1), name='f_pred')

    # get the expression of how we calculate the cost function
    # i.e. corss-entropy loss
    index = tensor.arange(x.shape[0])
    cost = -tensor.log(pred[index, y] + 1e-6).mean()                          

    return use_noise, x, y, f_pred_prob, f_pred, cost
    

