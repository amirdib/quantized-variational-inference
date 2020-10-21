import os
import pandas as pd
import numpy as np
from qvi.misc import uci
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from qvi.models.data import make_standardize_funs
from tensorflow import keras
from tensorflow.keras import layers

from qvi.misc.utils import split_to_nested_tensors

vb_dir   = os.path.dirname(__file__)
data_dir = os.path.join(vb_dir, "../../data/")


def trace_bnn(loss, grads, variables):
    return loss, tf.timestamp(), tf.concat(grads,axis=0)

def neural_network_batch(x,W_0,b_0,W_1,b_1):
    h = tf.nn.relu(tf.matmul(x, W_0) +  b_0[:,tf.newaxis,:])
    h = tf.matmul(h, W_1) + b_1[:,tf.newaxis,:]
    return h

def train_bnn_keras(units=30):
    data = pd.read_csv(vb_dir + '/../../data/metro.csv')
    data = data.drop(axis=1, columns=['date_time','holiday','weather_description','weather_main'])
    data.shape
    Xtrain, Ytrain = data.iloc[:,:-1], data.iloc[:,-1]
    target = tf.cast(Ytrain, dtype=tf.float32)
    features = tf.cast(Xtrain, dtype=tf.float32)
    n_features = features.shape[1]

    model = tf.keras.Sequential()
    model.add(layers.Input(n_features, name='input'))
    model.add(layers.Dense(units=units, activation='relu', name='HiddenLayer'))
    model.add(layers.Dense(units=1, name='Output'))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss='mse',       # mean squared error
                  metrics=['mae'])  # mean absolute error

    model.fit(features,target, epochs=20)
    true_weights = model.weights

    return model,target, features,  model.count_params(), true_weights
