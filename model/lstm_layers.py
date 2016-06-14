
import numpy as np
import theano
import theano.tensor as tensor
from utils import _p, numpy_floatX
from utils import ortho_weight, uniform_weight, zero_bias

""" Encoder using LSTM Recurrent Neural Network. """

def param_init_encoder(options, params, prefix='lstm_encoder'):
    
    n_x = options['n_x']
    n_h = options['n_h']
    
    W = np.concatenate([uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h)], axis=1)
    params[_p(prefix, 'W')] = W
    
    U = np.concatenate([ortho_weight(n_h),
                        ortho_weight(n_h),
                        ortho_weight(n_h),
                        ortho_weight(n_h)], axis=1)
    params[_p(prefix, 'U')] = U
    
    params[_p(prefix,'b')] = zero_bias(4*n_h)
    
    # It is observed that setting a high initial forget gate bias for LSTMs can 
    # give slighly better results (Le et al., 2015). Hence, the initial forget
    # gate bias is set to 3.
    params[_p(prefix, 'b')][n_h:2*n_h] = 3*np.ones((n_h,)).astype(theano.config.floatX)

    return params
    
def encoder(tparams, state_below, mask, seq_output=False, prefix='lstm_encoder'):
    
    """ state_below: size of  n_steps * n_samples * n_x
    """

    n_steps = state_below.shape[0]
    n_samples = state_below.shape[1]

    n_h = tparams[_p(prefix,'U')].shape[0]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
                    tparams[_p(prefix, 'b')]

    def _step(m_, x_, h_, c_, U):
        preact = tensor.dot(h_, U)
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, n_h))
        f = tensor.nnet.sigmoid(_slice(preact, 1, n_h))
        o = tensor.nnet.sigmoid(_slice(preact, 2, n_h))
        c = tensor.tanh(_slice(preact, 3, n_h))
        
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    seqs = [mask, state_below_]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                    n_samples,n_h),
                                              tensor.alloc(numpy_floatX(0.),
                                                    n_samples,n_h)],
                                non_sequences = [tparams[_p(prefix, 'U')]],
                                name=_p(prefix, '_layers'),
                                n_steps=n_steps,
                                strict=True)
    
    h_rval = rval[0] 
    if seq_output:
        return h_rval
    else:
        # size of n_samples * n_h
        return h_rval[-1]  
