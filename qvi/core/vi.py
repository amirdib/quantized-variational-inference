import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
import tensorflow as tf
from tensorflow_probability.python.vi import csiszar_divergence
tfd = tfp.distributions
from functools import partial
import matplotlib.pyplot as plt
from tensorflow_probability.python.internal import nest_util

from qvi.core.distribution import get_gaussian_quantization_weights
from qvi.misc.utils import split_to_nested_tensors


DTYPE = tf.float32
_reparameterized_elbo = partial(
    csiszar_divergence.monte_carlo_variational_loss,
    discrepancy_fn=csiszar_divergence.kl_reverse,
    use_reparameterization=True)

import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow_probability.python.vi import csiszar_divergence
tfd = tfp.distributions
from functools import partial
_reparameterized_elbo = partial(
    csiszar_divergence.monte_carlo_variational_loss,
    discrepancy_fn=csiszar_divergence.kl_reverse,
    use_reparameterization=True)

class VariationalInference:
    def __init__(self,
                 target_log_prob_fn,
                 surrogate_posterior, 
                 sample_size,
                 variational_loss_fn,
                 optimizer,
                 num_steps,
                 trace_fn=None,
                 name=''):
        self.target_log_prob_fn=target_log_prob_fn
        self.surrogate_posterior=surrogate_posterior
        self.trace_fn=trace_fn
        self.optimizer=optimizer
        self.num_steps=num_steps
        self.variational_loss_fn=variational_loss_fn
        self.trainable_variables=surrogate_posterior.trainable_variables
        self.sample_size=sample_size
        self.name = name
    
    def run(self):
        #pbar = tqdm(total=num_steps)
        self.trace = tfp.vi.fit_surrogate_posterior(
                    target_log_prob_fn=self.target_log_prob_fn,
                    surrogate_posterior=self.surrogate_posterior,
                    trace_fn=self.trace_fn,
                    optimizer=self.optimizer,
                    num_steps=self.num_steps,
                    variational_loss_fn=self.variational_loss_fn,
                    trainable_variables=self.trainable_variables,
                    sample_size=self.sample_size)
        
    def plot(self, abscissa='time',name=''):
        loss, timestamps, grads = self.trace
        if abscissa == 'time':
            x = timestamps - timestamps[0]
        elif abscissa == 'epochs':
            x = np.arange(0,len(loss))
        plt.plot(x, -loss, label=name)
        

class MCVariationalInference(VariationalInference):
    def __init__(self,
                 target_log_prob_fn,
                 surrogate_posterior, 
                 sample_size,
                 optimizer,
                 trace_fn,
                 num_steps,
                 name=''):
        super().__init__(target_log_prob_fn=target_log_prob_fn,
                 surrogate_posterior=surrogate_posterior, 
                 sample_size=sample_size,
                 variational_loss_fn=vi_mc,
                 optimizer=optimizer,
                 num_steps=num_steps,
                 trace_fn=trace_fn,
                 name='')

class RQMCVariationalInference(VariationalInference):
    def __init__(self,
                 target_log_prob_fn,
                 surrogate_posterior, 
                 sample_size,
                 optimizer,
                 trace_fn,
                 num_steps,
                 name=''):
        super().__init__(target_log_prob_fn=target_log_prob_fn,
                 surrogate_posterior=surrogate_posterior, 
                 sample_size=sample_size,
                 variational_loss_fn=vi_mc,
                 optimizer=optimizer,
                 num_steps=num_steps,
                 trace_fn=trace_fn,
                 name='')

        
class QuantizedVariationalInference(VariationalInference):
    def __init__(self,
                 target_log_prob_fn,
                 surrogate_posterior, 
                 sample_size,
                 optimizer,
                 trace_fn,
                 num_steps,
                 D,
                 name=''):
        
        self.D = D
        
        super().__init__(target_log_prob_fn=target_log_prob_fn,
                 surrogate_posterior=surrogate_posterior, 
                 sample_size=sample_size,
                 variational_loss_fn=partial(vi_quantized, seed=None, K='', D=self.D),
                 optimizer=optimizer,
                 num_steps=num_steps,
                 trace_fn=trace_fn,
                 name='')
        
class QuantizedRichardsonVariationalInference(VariationalInference):
    def __init__(self,
                 target_log_prob_fn,
                 surrogate_posterior, 
                 sample_size,
                 optimizer,
                 num_steps,
                 trace_fn,
                 D,
                 M,
                 name=''):
        
        self.D = D
        self.M = M
        
        super().__init__(target_log_prob_fn=target_log_prob_fn,
                 surrogate_posterior=surrogate_posterior, 
                 sample_size=sample_size,
                 variational_loss_fn=partial(vi_quantized_richardson, seed=None, D=self.D, M=self.M),
                 optimizer=optimizer,
                 num_steps=num_steps,
                 trace_fn=trace_fn,
                 name='')
        
def vi_quantized_richardson(target_log_prob_fn,
    surrogate_posterior,
    sample_size,
    seed,
    D,
    M):
    
    #N value is sample_size
    N = sample_size
    
    def q_divergence(sample_size):
        q_samples = surrogate_posterior.sample(sample_size)
        surrogate_posterior_log_prob = surrogate_posterior.log_prob(q_samples)
        target_log_prob = nest_util.call_fn(partial(target_log_prob_fn), q_samples)
        weights = get_gaussian_quantization_weights(shape= (sample_size,D), dtype=tf.float32)
        divergence = tfp.vi.kl_reverse(target_log_prob - surrogate_posterior_log_prob)
        return tf.tensordot(weights,divergence, axes=1)
    
    divN = tf.reduce_sum(q_divergence(N))
    divM = tf.reduce_sum(q_divergence(M))
    power = tf.constant(2.)
    coeff_pow = D

    reg_M = tf.math.pow(tf.cast(M,dtype=DTYPE),power/coeff_pow)
    reg_N = tf.math.pow(tf.cast(N,dtype=DTYPE),power/coeff_pow)
    elbo = ( reg_N * divN - reg_M * divM)/(reg_N - reg_M)
    return elbo


def vi_quantized(target_log_prob_fn,
    surrogate_posterior,
    sample_size,
    seed,
    D,
    K):
    q_samples = surrogate_posterior.sample(sample_size)
    
    surrogate_posterior_log_prob = surrogate_posterior.log_prob(q_samples)
    target_log_prob = nest_util.call_fn(partial(target_log_prob_fn), q_samples)
    weights = get_gaussian_quantization_weights(shape= (sample_size,D), dtype=tf.float32)
    divergence = tfp.vi.kl_reverse(target_log_prob - surrogate_posterior_log_prob)
    elbo =  tf.reduce_sum(tf.tensordot(weights,divergence, axes=1))

    #tf.print(elbo)
    return elbo

def vi_mc(target_log_prob_fn,
    surrogate_posterior,
    sample_size,
    seed=None):
    q_samples = surrogate_posterior.sample(sample_size)
    
    surrogate_posterior_log_prob = surrogate_posterior.log_prob(q_samples)
    target_log_prob = nest_util.call_fn(partial(target_log_prob_fn), q_samples)

    divergence = tfp.vi.kl_reverse(target_log_prob - surrogate_posterior_log_prob)
    elbo =  tf.reduce_mean(divergence, axis=0)
    return elbo

# def trace_bnn(loss, grads, variables):
#     pbar.set_description('ELBO: %s' % str(loss.numpy()))
#     pbar.update()
#     return loss, tf.timestamp(), grads

def build_meanfield_advi(jd_list, observed_node=-1, distribution=tfd.Normal, reinterpreted_batch_ndims_node = 1, **kwargs):
    """
      The inputted jointdistribution needs to be a batch version
    """
    list_of_values = jd_list.sample(1)  
    list_of_values.pop(observed_node)  
    distlist = []
    for i, value in enumerate(list_of_values):
        dtype = value.dtype
        rv_shape = value[0].shape
        #print(rv_shape)
        loc = tf.Variable(tf.zeros(rv_shape), 
            name='meanfield_%s_mu' % i,
            dtype=dtype)
        scale = tfp.util.TransformedVariable(tf.ones(rv_shape), tfb.Softplus(),
                                            name='meanfield_%s_scale' % i)
        
        approx_node = distribution(loc=loc,
                     scale=scale,
                    name='meanfield_%s' % i, **kwargs)
        
        if loc.shape == ():
            distlist.append(approx_node)
        else:
            distlist.append(
              tfd.Independent(approx_node, reinterpreted_batch_ndims=reinterpreted_batch_ndims_node)
            )

    meanfield_advi = tfd.JointDistributionSequential(distlist)
    return meanfield_advi

def trainable_normal_distribution(shape, name='', distribution=tfd.Normal, **kwargs):
    loc =   tf.Variable(tf.zeros(shape), name='{}_loc'.format(name))
    scale = tfp.util.TransformedVariable(tf.Variable(tf.fill(shape,1.), name='{}_scale'.format(name)),
                                         bijector = tfb.Softplus())
    return distribution(loc, scale, name=name, **kwargs)
