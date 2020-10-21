from qvi.core.vi import trainable_normal_distribution
from qvi.core.distribution import QuantizedNormal, QMCNormal
import pandas as pd
from tqdm import tqdm

import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
tfd = tfp.distributions

DTYPE = tf.float32


def blr_conditioned_log_prob_fn(data):
    y = tf.cast(data.iloc[:, -1].values, dtype=DTYPE)
    X = tf.cast(data.iloc[:, :-1].values, dtype=DTYPE)
    D = X.shape[-1]

    def bayesian_regression(features, weights):
        """Creates a joint distribution representing our generative process."""
        return tfd.JointDistributionSequential([
            tfd.Independent(
                tfd.Normal(tf.zeros(D+1), 1.),
                1),
            lambda weights:tfd.Independent(
                tfd.Normal(loc=tf.tensordot(
                    features, weights[:-1], [[1], [0]]) + weights[-1], scale=2.),
                1)
        ])

    def conditioned_log_prob_one_sample(weight):
        return bayesian_regression(X, weight).log_prob([weight, y])

    return lambda weights: tf.stack(
        list(map(
            lambda weight: conditioned_log_prob_one_sample(weight),
            tf.unstack(weights)))
    )


def build_quantized_br_posterior(D):
    return tfd.Independent(trainable_normal_distribution(shape=[D], name='weights', distribution=QuantizedNormal), 1)

def build_qmc_br_posterior(D):
    return tfd.Independent(trainable_normal_distribution(shape=[D], name='weights', distribution=QMCNormal, randomized=False), 1)

def build_rqmc_br_posterior(D):
    return tfd.Independent(trainable_normal_distribution(shape=[D], name='weights', distribution=QMCNormal, randomized=True), 1)


def build_qmcoffline_br_posterior(D):
    return tfd.Independent(trainable_normal_distribution(shape=[D], name='weights', distribution=QMCNormalOffline, randomized=True), 1)

def build_mc_br_posterior(D):
    return tfd.Independent(trainable_normal_distribution(shape=[D], name='weights', distribution=tfd.Normal), 1)

# def build_qmcrtb_br_posterior(D):
#     return tfd.Independent(trainable_normal_distribution(shape=[D], name='weights', 
#                                                          distribution=QMCNormalRTB), 1)
