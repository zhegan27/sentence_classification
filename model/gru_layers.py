
import numpy as np
import theano
import theano.tensor as tensor
from utils import _p, numpy_floatX
from utils import ortho_weight, uniform_weight, zero_bias

""" Encoder using GRU Recurrent Neural Network. """

def param_init_encoder(options, params, prefix='gru_encoder'):
    
    n_x = options['n_x']
    n_h = options['n_h']
    
    W = np.concatenate([uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h)], axis=1)
    params[_p(prefix,'W')] = W
    
    U = np.concatenate([ortho_weight(n_h),
                        ortho_weight(n_h)], axis=1)
    params[_p(prefix,'U')] = U
    
    params[_p(prefix,'b')] = zero_bias(2*n_h)

    Wx = uniform_weight(n_x, n_h)
    params[_p(prefix,'Wx')] = Wx
    
    Ux = ortho_weight(n_h)
    params[_p(prefix,'Ux')] = Ux
    
    params[_p(prefix,'bx')] = zero_bias(n_h)

    return params
    

def encoder(tparams, state_below, mask, seq_output=False, prefix='gru_encoder'):
    
    """ state_below: size of n_steps * n_samples * n_x 
    """

    n_steps = state_below.shape[0]
    n_samples = state_below.shape[1]

    n_h = tparams[_p(prefix,'Ux')].shape[1]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
                    tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
                    tparams[_p(prefix, 'bx')]

    def _step(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, n_h))
        u = tensor.nnet.sigmoid(_slice(preact, 1, n_h))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info = [tensor.alloc(numpy_floatX(0.),
                                                             n_samples, n_h)],
                                non_sequences = [tparams[_p(prefix, 'U')],
                                                 tparams[_p(prefix, 'Ux')]],
                                name=_p(prefix, '_layers'),
                                n_steps=n_steps,
                                strict=True)
    if seq_output:
        return rval
    else:
        # size of n_samples * n_h
        return rval[-1]  
