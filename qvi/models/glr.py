import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
tfd = tfp.distributions

from qvi.core.distribution import QuantizedNormal


DTYPE = tf.float32


def compute_co_variates(dataframe):
    dataframe = dataframe.sort_values(['eth', 'precinct'])
    nep = np.log(dataframe['past.arrests'].values) + np.log(15./12)
    yep = dataframe['stops'].values
    return nep, yep


def compute_target_log_prob_frisk_fn(dataframe):
    nep, yep = compute_co_variates(dataframe)
    yep = tf.convert_to_tensor(yep, dtype=DTYPE)
    nep = tf.convert_to_tensor(nep, dtype=DTYPE)
    Np = tf.cast(32, dtype=tf.int32)
    Ne = tf.cast(3, dtype=tf.int32)
    var_name = ['alpha', 'beta', 'e', 'p', 'mu', 'y']

    def glm(mu, p, e): return tfd.Independent(tfd.Poisson(
        log_rate=mu[..., tf.newaxis] + log_rate(e, p) + tf.cast(nep, dtype=DTYPE)), 1)

    def swap_first_last_axis(tensor):
        return tf.transpose(tensor, perm=list(range(len(tensor.shape))[1:]) + [0])

    def log_rate(p, e, verbose=False):
        if verbose:
            print('\n #################################################')
            print('\n ################### Debug #######################')
            print('\n #################################################\n')
            print('p shape'.format(p.shape))
            print('e shape'.format(e.shape))
            print(p[..., :, tf.newaxis] + e[..., tf.newaxis, :])
        if len(p.shape) == 1:
            reshape = p.shape[0]*e.shape[0]
        else:
            reshape = list(e.shape[:-1]) + [-1]
        return tf.reshape(e[..., :, tf.newaxis] + p[..., tf.newaxis, :], shape=reshape)

    def log_rate(x, y, verbose=False):
        matrix_mul_vector = []
        for i in range(Ne):
            for j in range(Np):
                matrix_mul_vector.append(x[..., j] + y[..., i])
        rate = tf.convert_to_tensor(matrix_mul_vector)
        return swap_first_last_axis(rate)

    def pooled_model():
        """Creates a joint distribution representing our generative process."""
        return tfd.JointDistributionSequential([
            tfd.Sample(tfd.Normal(loc=0, scale=1.), sample_shape=Ne),  # salpha
            tfd.Sample(tfd.Normal(loc=0., scale=1), sample_shape=Np),  # sbeta
            lambda beta: tfd.Independent(tfd.Normal(
                loc=0., scale=tf.math.exp(beta)), 1),  # e
            lambda e, beta,  alpha: tfd.Independent(
                tfd.Normal(loc=0., scale=tf.math.exp(alpha)), 1),  # p
            tfd.Normal(loc=0., scale=1.),  # mu
            lambda mu, p, e: glm(mu, p, e)
        ])

    def conditioned_log_prob(alpha, beta, e, p, mu, func_prob='log_prob'):
        return pooled_model().__getattribute__(func_prob)([alpha, beta, e, p, mu, yep])

    return conditioned_log_prob, pooled_model
