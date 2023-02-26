"""
File Name: nn.py
Purpose: The main code for the feedforward networks assignment. See README.md for details.
Author(s): Rey Sanayei
Version: 1.0 (9/04/2022)
"""
from typing import Tuple, Dict

import tensorflow as tf


def create_auto_mpg_deep_and_wide_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tf.keras.models.Model,
                                                tf.keras.models.Model]:
    """Creates one deep neural network and one wide neural network.
    The networks should have the same (or very close to the same) number of
    parameters and the same activation functions.

    The neural networks will be asked to predict the number of miles per gallon
    that different cars get. They will be trained and tested on the Auto MPG
    dataset from:
    https://archive.ics.uci.edu/ml/datasets/auto+mpg

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (deep neural network, wide neural network)
    """

    deep_model = tf.keras.models.Sequential()
    deep_model.add(tf.keras.layers.Dense(14, activation=tf.keras.activations.relu,
                                         input_shape=[n_inputs]))
    deep_model.add(tf.keras.layers.Dense(14, activation=tf.keras.activations.relu))
    deep_model.add(tf.keras.layers.Dense(14, activation=tf.keras.activations.relu))
    deep_model.add(tf.keras.layers.Dense(n_outputs))
    optimizer = tf.keras.optimizers.RMSprop(0.1)
    deep_model.compile(loss='mean_squared_error',
                       optimizer=optimizer,
                       metrics=['mean_absolute_error', 'mean_squared_error'])

    wide_model = tf.keras.models.Sequential()
    wide_model.add(tf.keras.layers.Dense(20, activation=tf.keras.activations.relu,
                                         input_shape=[n_inputs]))
    wide_model.add(tf.keras.layers.Dense(20, activation=tf.keras.activations.relu))
    wide_model.add(tf.keras.layers.Dense(n_outputs))
    wide_model.compile(loss='mean_squared_error',
                       optimizer=optimizer,
                       metrics=['mean_absolute_error', 'mean_squared_error'])
    res_tuple = (deep_model, wide_model)
    return res_tuple


def create_delicious_relu_vs_tanh_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tf.keras.models.Model,
                                                tf.keras.models.Model]:
    """Creates one neural network where all hidden layers have ReLU activations,
    and one where all hidden layers have tanh activations. The networks should
    be identical other than the difference in activation functions.

    The neural networks will be asked to predict the 0 or more tags associated
    with a del.icio.us bookmark. They will be trained and tested on the
    del.icio.us dataset from:
    https://github.com/dhruvramani/Multilabel-Classification-Datasets
    which is a slightly simplified version of:
    https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi-Instance+Learning+with+Instance+Labels

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (ReLU neural network, tanh neural network)
    """

    relu_model = tf.keras.models.Sequential()
    relu_model.add(
        tf.keras.layers.Dense(14,
                              activation=tf.keras.activations.relu,
                              input_shape=[n_inputs]))
    relu_model.add(tf.keras.layers.Dense(14,
                                         activation=tf.keras.activations.relu))
    relu_model.add(tf.keras.layers.Dense(14,
                                         activation=tf.keras.activations.relu))
    relu_model.add(tf.keras.layers.Dense(n_outputs,
                                         activation=tf.keras.activations.sigmoid))
    optimizer = tf.keras.optimizers.RMSprop(0.1)
    relu_model.compile(loss='hinge',
                       optimizer=optimizer,
                       metrics=['hinge'])

    tanh_model = tf.keras.models.Sequential()
    tanh_model.add(
        tf.keras.layers.Dense(14,
                              activation=tf.keras.activations.tanh,
                              input_shape=[n_inputs]))
    tanh_model.add(tf.keras.layers.Dense(14,
                                         activation=tf.keras.activations.tanh))
    tanh_model.add(tf.keras.layers.Dense(14,
                                         activation=tf.keras.activations.tanh))
    tanh_model.add(tf.keras.layers.Dense(n_outputs,
                                         activation=tf.keras.activations.sigmoid))
    optimizer = tf.keras.optimizers.RMSprop(0.1)
    tanh_model.compile(loss='hinge',
                       optimizer=optimizer,
                       metrics=['hinge'])
    res_tuple = (relu_model, tanh_model)
    return res_tuple


def create_activity_dropout_and_nodropout_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tf.keras.models.Model,
                                                tf.keras.models.Model]:
    """Creates one neural network with dropout applied after each layer, and
    one neural network without dropout. The networks should be identical other
    than the presence or absence of dropout.

    The neural networks will be asked to predict which one of six activity types
    a smartphone user was performing. They will be trained and tested on the
    UCI-HAR dataset from:
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (dropout neural network, no-dropout neural network)
    """
    dropout_model = tf.keras.models.Sequential()
    dropout_model.add(
        tf.keras.layers.Dense(100,
                              activation=tf.keras.activations.relu,
                              input_shape=[n_inputs]))
    dropout_model.add(tf.keras.layers.Dropout(0.2))
    dropout_model.add(tf.keras.layers.Dense(100,
                                            activation=tf.keras.activations.relu))
    dropout_model.add(tf.keras.layers.Dropout(0.2))
    dropout_model.add(tf.keras.layers.Dense(100,
                                            activation=tf.keras.activations.relu))
    dropout_model.add(tf.keras.layers.Dropout(0.2))
    dropout_model.add(tf.keras.layers.Dense(n_outputs,
                                            activation=tf.keras.activations.softmax))
    optimizer = tf.keras.optimizers.Adam(0.001)
    dropout_model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

    non_dropout_model = tf.keras.models.Sequential()
    non_dropout_model.add(
        tf.keras.layers.Dense(100,
                              activation=tf.keras.activations.relu,
                              input_shape=[n_inputs]))
    non_dropout_model.add(tf.keras.layers.Dense(100,
                                                activation=tf.keras.activations.relu))
    non_dropout_model.add(tf.keras.layers.Dense(100,
                                                activation=tf.keras.activations.relu))

    non_dropout_model.add(tf.keras.layers.Dense(n_outputs,
                                                activation=tf.keras.activations.softmax))
    optimizer = tf.keras.optimizers.Adam(0.001)
    non_dropout_model.compile(loss='categorical_crossentropy',
                              optimizer=optimizer,
                              metrics=['accuracy'])
    res_tuple = (dropout_model, non_dropout_model)
    return res_tuple


def create_income_earlystopping_and_noearlystopping_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tf.keras.models.Model,
                                                Dict,
                                                tf.keras.models.Model,
                                                Dict]:
    """Creates one neural network that uses early stopping during training, and
    one that does not. The networks should be identical other than the presence
    or absence of early stopping.

    The neural networks will be asked to predict whether a person makes more
    than $50K per year. They will be trained and tested on the "adult" dataset
    from:
    https://archive.ics.uci.edu/ml/datasets/adult

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (
        early-stopping neural network,
        early-stopping parameters that should be passed to Model.fit,
        no-early-stopping neural network,
        no-early-stopping parameters that should be passed to Model.fit
    )
    """
    early_stopping_model = tf.keras.models.Sequential()
    early_stopping_model.add(
        tf.keras.layers.Dense(14,
                              activation=tf.keras.activations.tanh,
                              input_shape=[n_inputs]))
    early_stopping_model.add(tf.keras.layers.Dense(14,
                                                   activation=tf.keras.activations.tanh))
    early_stopping_model.add(tf.keras.layers.Dense(14,
                                                   activation=tf.keras.activations.tanh))
    early_stopping_model.add(tf.keras.layers.Dense(n_outputs,
                                                   activation=tf.keras.activations.sigmoid))
    optimizer = tf.keras.optimizers.Adam(0.001)
    early_stopping_model.compile(loss='hinge',
                                 optimizer=optimizer,
                                 metrics=['hinge'])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=3,
        verbose=0,
        mode='min',
        baseline=None,
        restore_best_weights=True
    )
    early_stop_dict = {'callbacks': early_stop}

    non_early_stopping_model = tf.keras.models.Sequential()
    non_early_stopping_model.add(
        tf.keras.layers.Dense(14,
                              activation=tf.keras.activations.tanh,
                              input_shape=[n_inputs]))
    non_early_stopping_model.add(tf.keras.layers.Dense(14,
                                                       activation=tf.keras.activations.tanh))
    non_early_stopping_model.add(tf.keras.layers.Dense(14,
                                                       activation=tf.keras.activations.tanh))
    non_early_stopping_model.add(
        tf.keras.layers.Dense(n_outputs,
                              activation=tf.keras.activations.sigmoid))
    optimizer = tf.keras.optimizers.Adam(0.001)
    non_early_stopping_model.compile(loss='hinge',
                                     optimizer=optimizer,
                                     metrics=['hinge'])

    non_early_stop_dict = {}

    res_tuple = (early_stopping_model, early_stop_dict,
                 non_early_stopping_model, non_early_stop_dict)
    return res_tuple
