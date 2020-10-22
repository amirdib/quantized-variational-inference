from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import nest_util
from tensorflow_probability import bijectors as tfb
import tensorflow as tf
from tensorflow_probability.python.vi import csiszar_divergence
tfd = tfp.distributions

from qvi.core.vi import *
from qvi.misc.utils import split_to_nested_tensors
from qvi.core.distribution import get_gaussian_quantization_weights

class Experiment():

    def __init__(self,
                 conditioned_log_prob,
                 optimizer,
                 num_steps,
                 trace_fn,
                 mc_surrogate_builder,
                 q_surrogate_builder,
                 D,
                 M,
                 qmc_surrogate_builder,
                 rqmc_surrogate_builder,
                 rq_surrogate_builder,
                 sample_size,
                 optimizer_params,
                 type_experiment='simple'):

        self.conditioned_log_prob = conditioned_log_prob
        self.mc_surrogate_builder = mc_surrogate_builder
        self.qmc_surrogate_builder = qmc_surrogate_builder
        self.q_surrogate_builder = q_surrogate_builder
        self.rq_surrogate_builder = rq_surrogate_builder
        self.rqmc_surrogate_builder = rqmc_surrogate_builder
        self.optimizer_params = optimizer_params
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.trace_fn = trace_fn
        self.D = D
        self.M = M
        if type_experiment == 'sample_size':
            self.sample_size_r = sample_size
            self.sample_size = sample_size + M
        else:
            self.sample_size = sample_size
            self.sample_size_r = sample_size

    def run(self, N_RUNS):

#=================================== MC =========================================
        if self.mc_surrogate_builder is not None:
            surrogate_posterior = self.mc_surrogate_builder(self.D)
            mc = MCVariationalInference(target_log_prob_fn=self.conditioned_log_prob,
                                        surrogate_posterior=surrogate_posterior,
                                        optimizer=self.optimizer(
                                            **self.optimizer_params),
                                        trace_fn=self.trace_fn,
                                        num_steps=self.num_steps,
                                        sample_size=self.sample_size,
                                        name='MC VI')
            mc.run()
            self.mcvitrace = mc.trace

        if self.mc_surrogate_builder is not None:
            mctraces = multiple_runs(target_log_prob_fn=self.conditioned_log_prob,
                                     surrogate_posterior_builder=self.mc_surrogate_builder,
                                     optimizer=self.optimizer,
                                     optimizer_params=self.optimizer_params,
                                     num_steps=self.num_steps,
                                     trace_fn=self.trace_fn,
                                     D=self.D,
                                     n_visamples=N_RUNS,
                                     sample_size=self.sample_size)
            self.mcvitraces = mctraces
#=================================== QMC =========================================

        if self.qmc_surrogate_builder is not None:
            surrogate_posterior = self.qmc_surrogate_builder(self.D)
            qmcvi = MCVariationalInference(target_log_prob_fn=self.conditioned_log_prob,
                                           surrogate_posterior=surrogate_posterior,
                                           optimizer=self.optimizer(
                                               **self.optimizer_params),
                                           trace_fn=self.trace_fn,
                                           num_steps=self.num_steps,
                                           sample_size=self.sample_size,
                                           name='Quasi Monte Carlo VI')
            qmcvi.run()
            self.qmcvitraces = qmcvi.trace


        if self.rqmc_surrogate_builder is not None:
            surrogate_posterior = self.rqmc_surrogate_builder(self.D)
            rqmcvi = MCVariationalInference(target_log_prob_fn=self.conditioned_log_prob,
                                            surrogate_posterior=surrogate_posterior,
                                            optimizer=self.optimizer(
                                                **self.optimizer_params),
                                            trace_fn=self.trace_fn,
                                            num_steps=self.num_steps,
                                            sample_size=self.sample_size,
                                            name='Quasi Monte Carlo VI')
            rqmcvi.run()
            self.rqmcvitrace = rqmcvi.trace

        if self.rqmc_surrogate_builder is not None:
            rqmctraces = multiple_runs(target_log_prob_fn=self.conditioned_log_prob,
                                       surrogate_posterior_builder=self.rqmc_surrogate_builder,
                                       optimizer=self.optimizer,
                                       optimizer_params=self.optimizer_params,
                                       num_steps=self.num_steps,
                                       trace_fn=self.trace_fn,
                                       D=self.D,
                                       n_visamples=N_RUNS,
                                       sample_size=self.sample_size)
            self.rqmcvitraces = rqmctraces

#=================================== QVI =========================================

        if self.q_surrogate_builder is not None:
            surrogate_posterior = self.q_surrogate_builder(self.D)
            qvi = QuantizedVariationalInference(target_log_prob_fn=self.conditioned_log_prob,
                                                surrogate_posterior=surrogate_posterior,
                                                optimizer=self.optimizer(
                                                    **self.optimizer_params),
                                                trace_fn=self.trace_fn,
                                                num_steps=self.num_steps,
                                                sample_size=self.sample_size,
                                                D=self.D,
                                                name='Quantized VI')
            qvi.run()
            self.qvitraces = qvi.trace

        if self.rq_surrogate_builder is not None:
            surrogate_posterior = self.rq_surrogate_builder(self.D)
            rqvi = QuantizedRichardsonVariationalInference(
                target_log_prob_fn=self.conditioned_log_prob,
                surrogate_posterior=surrogate_posterior,
                optimizer=self.optimizer(**self.optimizer_params),
                trace_fn=self.trace_fn,
                num_steps=self.num_steps,
                sample_size=self.sample_size_r,
                M=self.M,
                D=self.D ,
                name='Quantized Richardson VI')
            rqvi.run()
            self.rqvitraces = rqvi.trace

        return None

def multiple_runs(target_log_prob_fn,
                  surrogate_posterior_builder,
                  optimizer,
                  optimizer_params,
                  trace_fn,
                  num_steps,
                  sample_size,
                  D,
                  n_visamples=10):
    traces = list()
    for i in tqdm(range(n_visamples)):
        surrogate_posterior = surrogate_posterior_builder(D)
        mcvi = MCVariationalInference(target_log_prob_fn=target_log_prob_fn,
                                      surrogate_posterior=surrogate_posterior,
                                      optimizer=optimizer(**optimizer_params),
                                      trace_fn=trace_fn,
                                      num_steps=num_steps,
                                      sample_size=sample_size,
                                      name='Monte Carlo VI')
        mcvi.run()
        traces.append(mcvi.trace)
    return traces


def compute_traces_from_multiple_trainning(traces):
    losses = tf.stack(list(map(lambda t: t[0], traces)), axis=1)
    timestamps = tf.reduce_mean(
        tf.stack(list(map(lambda t: t[1], traces)), axis=1), axis=-1)
    gradients = tf.reduce_sum(
        tf.square(tf.stack(list(map(lambda t: t[2], traces)), axis=1)), axis=-1)
    return losses, timestamps, gradients
