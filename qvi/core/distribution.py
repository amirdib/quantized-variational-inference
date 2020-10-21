#import rpy2.robjects.packages as rpackages
import pickle
import random
from pathlib import Path
import qvi
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions


# randtoolbox = rpackages.importr('randtoolbox')

DTYPE = tf.float32
DATA_PATH = str(Path(__file__).parent.parent.parent / 'data')


class QMCNormal(tfd.Normal):
    """The Quantized Normal distribution with location `loc` and `scale` parameters.

    ```python
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    # Define a single scalar Normal distribution.
    dist = tfd.Normal(loc=0., scale=3.)
    ```
    """

    def __init__(self,
                 loc,
                 scale,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='QMCNormal',
                 randomized=True):
        """Construct Quasi Monte Carlo Normal distributions with mean and stddev `loc` and `scale`.
        The parameters `loc` and `scale` must be shaped in a way that supports
        broadcasting (e.g. `loc + scale` is a valid operation).
        Args:
        loc: Floating point tensor; the means of the distribution(s).
        scale: Floating point tensor; the stddevs of the distribution(s).
        Must contain only positive values.
        validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        name: Python `str` name prefixed to Ops created by this class.
        randomized: if QMC sequences should be randomized.
        Raises: TypeError: if `loc` and `scale` have different `dtype`.
        """
        super().__init__(loc,
                         scale,
                         validate_args=False,
                         allow_nan_stats=True,
                         name=name)
        self.weights = None
        self.randomized = randomized

    def _sample_n(self, n, seed=None):

        batch_shape = self._batch_shape_tensor(loc=self.loc, scale=self.scale)
        shape = tf.concat([[n], batch_shape],
                          axis=0)
        # Standard Quantile Function for Normal Distribution
        std_norm_ppf = tfd.Normal(0., 1.).quantile
        if shape.shape == 1:
            samples = tfp.mcmc.sample_halton_sequence(
                dim=1,
                num_results=n, randomized=self.randomized)
        elif shape.shape == 2:

            dim = batch_shape[-1]

            samples = tfp.mcmc.sample_halton_sequence(
                dim=dim,
                num_results=n, randomized=self.randomized)

            self.halton = tf.cast(samples, dtype=DTYPE)
        else:
            raise ValueError('Must feed correct shape value')
        samples = tf.cast(tf.reshape(
            std_norm_ppf(samples), shape), dtype=DTYPE)
        return self.loc + samples * self.scale


class QuantizedNormal(tfd.Normal):
    """The Quantized Normal distribution with location `loc` and `scale` parameters.
    ```python
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    # Define a single scalar Normal distribution.
    dist = tfd.Normal(loc=0., scale=3.)
    ```
    """

    def __init__(self,
                 loc,
                 scale,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='Quantized Normal'):
        super().__init__(loc,
                         scale,
                         validate_args=False,
                         allow_nan_stats=True,
                         name=name)
        self.weights = None
        """Construct Quasi Monte Carlo Normal distributions with mean and stddev `loc` and `scale`.
        The parameters `loc` and `scale` must be shaped in a way that supports
        broadcasting (e.g. `loc + scale` is a valid operation).
        Args:
        loc: Floating point tensor; the means of the distribution(s).
        scale: Floating point tensor; the stddevs of the distribution(s).
        Must contain only positive values.
        validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        name: Python `str` name prefixed to Ops created by this class.
        randomized: if QMC sequences should be randomized.
        Raises: TypeError: if `loc` and `scale` have different `dtype`.
        """

    def get_weights(self):
        return self.weights

    def _sample_n(self, n, seed=None, K=''):

        batch_shape = self._batch_shape_tensor(loc=self.loc, scale=self.scale)
        shape = tf.concat([[n], batch_shape],
                          axis=0)

        if shape.shape == 1:
            grid = get_gaussian_quantization(N=n, d=1, K=K)
            samples, weights = tf.cast(
                grid[0], self.dtype), tf.cast(grid[1], self.dtype)
            samples = tf.reshape(samples, shape)
        elif shape.shape == 2:
            dim = batch_shape[-1]
            grid = get_gaussian_quantization(N=n, d=dim, K=K)
            samples, weights = tf.cast(
                grid[0], self.dtype), tf.cast(grid[1], self.dtype)
        else:
            dim = batch_shape[-1]
            grid = get_gaussian_quantization(N=n, d=dim, K=K)
            samples, weights = tf.cast(
                grid[0], self.dtype), tf.cast(grid[1], self.dtype)
            samples = tf.repeat(samples[:, np.newaxis, ...], shape[1], axis=1)

        self.weights = tf.cast(weights, self.dtype)
        return self.loc + samples * self.scale


# def generate_sobolov(n, dim, seed=1223, init=False):
#     return tf.cast(np.array(randtoolbox.sobol(n=int(n),
#                                               dim=int(dim), init=init,
#                                               scrambling=1, seed=int(seed))),
#                    dtype=DTYPE)


# class QMCNormalRTB(tfd.Normal):
#     def __init__(self,
#                  loc,
#                  scale,
#                  validate_args=False,
#                  allow_nan_stats=True,
#                  name='Normal',
#                  randomized=True):
#         super().__init__(loc,
#                          scale,
#                          validate_args=False,
#                          allow_nan_stats=True,
#                          name=name)
#         self.weights = None
#         self.randomized = randomized

#     def _sample_n(self, n, seed=None):

#         batch_shape = self._batch_shape_tensor(loc=self.loc, scale=self.scale)
#         shape = tf.concat([[n], batch_shape],
#                           axis=0)

#         # Standard Quantile Function for Normal Distribution
#         std_norm_ppf = tfd.Normal(0., 1.).quantile
#         # seed = tf.random.uniform(
#         #     [], minval=1, maxval=int(1e9), dtype=tf.dtypes.int32)

#         # if shape.shape == 1:
#         #     samples = tf.py_function(generate_sobolov, inp=[
#         #                              n, 1, seed], Tout=tf.float32)
#         # elif shape.shape == 2:
#         #     dim = batch_shape[-1]
#         #     samples = tf.py_function(generate_sobolov, inp=[
#         #                              n, dim, seed], Tout=tf.float32)

#         if shape.shape == 1:
#             samples = generate_sobolov(n, 1)
#         elif shape.shape == 2:
#             dim = batch_shape[-1]
#             samples = generate_sobolov(n, dim)
#         else:
#             raise ValueError('Must feed correct shape value')

#         samples = tf.cast(tf.reshape(
#             std_norm_ppf(samples), shape), dtype=DTYPE)
#         return self.loc + samples * self.scale

# class NormalRTB(tfd.Normal):
#     def __init__(self,
#                  loc,
#                  scale,
#                  validate_args=False,
#                  allow_nan_stats=True,
#                  name='Normal',
#                  randomized=True):
#         super().__init__(loc,
#                          scale,
#                          validate_args=False,
#                          allow_nan_stats=True,
#                          name=name)
#         self.weights = None
#         self.randomized = randomized

#     def _sample_n(self, n, seed=None):

#         batch_shape = self._batch_shape_tensor(loc=self.loc, scale=self.scale)
#         shape = tf.concat([[n], batch_shape],
#                           axis=0)

#         seed = tf.random.uniform(
#             [], minval=1, maxval=int(1e9), dtype=tf.dtypes.int32)

#         if shape.shape == 1:
#             samples = tf.py_function(generate_sobolov, inp=[
#                                      n, 1, seed], Tout=tf.float32)
#         elif shape.shape == 2:
#             dim = batch_shape[-1]
#             samples = tf.py_function(generate_sobolov, inp=[
#                                      n, dim, seed], Tout=tf.float32)
#         else:
#             raise ValueError('Must feed correct shape value')
#         samples = tf.cast(tf.reshape(
#             std_norm_ppf(samples), shape), dtype=DTYPE)
#         return self.loc + samples * self.scale


def get_gaussian_quantization(N, d=2, K=''):
    # print('###Grid### {} {}'.format(N,d))
    grid = np.loadtxt(DATA_PATH + '/grids/{}_{}_nopti{}'.format(N, d, K))

    values = grid[:-1, 1:d+1]
    voronoi_weights = grid[:-1, 0]
    return values, voronoi_weights


def get_qmc_samples(N, d=2, dtype=tf.float32):
    samples = np.loadtxt(DATA_PATH + '/grids/qmc_{}_{}'.format(N, d))
    return tf.cast(samples, tf.float32)


def get_gaussian_quantization_weights(shape, K='', dtype=tf.float32):
    n = shape[0]
    if len(shape) == 1:
        return tf.cast(np.loadtxt(DATA_PATH + '/grids/{N}_{d}_nopti{K}'.format(N=n, d=1, K=K))[:-1, 0], dtype)
    elif len(shape) == 2:
        dim = shape[-1]
        return tf.cast(np.loadtxt(DATA_PATH + '/grids/{N}_{d}_nopti{K}'.format(N=n, d=dim, K=K))[:-1, 0], dtype)
