import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions

import dill as pickle

def flatten_weights(weights):
    return tf.concat(list(map(lambda x: tf.reshape(x, -1), tf.nest.flatten(model.weights))), axis=0)


def split_to_nested_tensors(tensor, shape_structure, axis=1):
    '''Split a tensor along axis and reshape to nested structure shape_structure'''

    prod_struct = list(map(tf.reduce_prod, shape_structure))
    split_prod = tf.split(tensor, prod_struct, axis=axis)
    return [tf.reshape(layer, [-1] + list(struct)) for layer, struct
            in zip(split_prod, shape_structure)]


def pprint_method_name(method_name):
    print(
        '---------------------- {method_name} ----------------------'.format(method_name=method_name))


def expend_dim_cond(tensor):
    tensor = tf.cond(tf.equal(tf.rank(tensor), 1),
                     lambda: tensor,
                     lambda: tf.expand_dims(tensor, -1),
                     )
    return tensor


def apply_expdim(tensor):
    return list(map(expend_dim_cond, tensor))


def trainable_normal_distribution(shape, name='', distribution=tfd.Normal):
    loc = tf.Variable(tf.zeros(shape), name='{}_loc'.format(name))
    scale = tf.Variable(tf.fill(shape, 2.), name='{}_scale'.format(name))
    return distribution(loc, scale, name=name)


def extract_from_samples(states, var_name):
    return {name: state for name, state in zip(var_name, states)}

def write_pickle(data, path, **kwargs):
    with open(path, 'wb') as output_file:
        pickle.dump(data, output_file, protocol=4)
def read_pickle(path):
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)
