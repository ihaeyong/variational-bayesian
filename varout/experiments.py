#!/usr/bin/env python

# Going by the architecture used in Srivastava's paper, detailed here:
# https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/papers/dropout/mnist_valid.yaml
# 
# Unsure how to deal with input dropout; we're assuming (in the case of Wang)
# that input dropout is propagated through to be noise on the pre-nonlinearity
# activations. But then, the final noise is going to end up directly damaging
# the predictions, which makes no sense. Probably better just to ignore reason
# in this case and just write the architecture the way it's probably supposed
# to be.

from varout import layers
import holonets
import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates
import theano
import theano.tensor as T
# import urllib2
import urllib.request
import pickle as pickle

import imp
import argparse
from collections import OrderedDict
import sys
import os
import numpy as np
import time
def conv_bn_rectify(net, num_filters):
    net = lasagne.layers.Conv2DLayer(net, int(num_filters), 3, W=lasagne.init.Normal(), pad=1, nonlinearity=lasagne.nonlinearities.linear)
    net = lasagne.layers.BatchNormLayer(net, epsilon=1e-3)
    net = lasagne.layers.NonlinearityLayer(net)

    return net
def wangDropoutArchitecture(batch_size=1000, input_dim=784, output_dim=10,
                            DropoutLayer=layers.WangGaussianDropout,
                            n_hidden=100):
    l_in = lasagne.layers.InputLayer((batch_size, input_dim))
    l_hidden_1 = lasagne.layers.DenseLayer(l_in, num_units=n_hidden, 
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_1 = DropoutLayer(l_hidden_1, p=0.2)
    l_hidden_2 = lasagne.layers.DenseLayer(l_drop_1, num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_2 = DropoutLayer(l_hidden_2, p=0.5)
    l_out = lasagne.layers.DenseLayer(l_drop_2, num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax)
    l_drop_3 = DropoutLayer(l_out, p=0.5)
    return l_drop_3

def wangDropoutArchitecture_cifar(batch_size=1000, output_dim=10,
                            DropoutLayer=layers.WangGaussianDropout,
                            n_hidden=1):
    input_x = T.tensor4("input")
    l_in = lasagne.layers.InputLayer((batch_size, 3, 32, 32), input_x)
    l_hidden_1 = lasagne.layers.Conv2DLayer(l_in, num_filters = int(32*n_hidden),  filter_size=3, stride=(2,2),nonlinearity=lasagne.nonlinearities.softplus)
    l_drop_1 = DropoutLayer(l_hidden_1, p=0.5)

    l_hidden_2 = lasagne.layers.Conv2DLayer(l_drop_1, num_filters = int(64*n_hidden),  filter_size=3, stride=(2,2),nonlinearity=lasagne.nonlinearities.softplus)
    l_drop_2 = DropoutLayer(l_hidden_2, p=0.5)

    # l_hidden_3 = lasagne.layers.DenseLayer(l_drop_2, num_units=int(128 * n_hidden),nonlinearity=lasagne.nonlinearities.rectify)
    l_hidden_3 = lasagne.layers.DenseLayer(l_drop_2, num_units=int(128 * n_hidden), nonlinearity=None)
    l_drop_3 = DropoutLayer(l_hidden_3, p=0.5)
    #l_hidden_4 = lasagne.layers.DenseLayer(l_drop_3, num_units=int(128 * n_hidden), nonlinearity=lasagne.nonlinearities.rectify)
    l_hidden_4 = lasagne.layers.DenseLayer(l_drop_3, num_units=int(128 * n_hidden), nonlinearity=None)
    l_drop_4 = DropoutLayer(l_hidden_4, p=0.5)
    l_out = lasagne.layers.DenseLayer(l_drop_4, num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out



def vardropBDropoutArchitecture(batch_size=1000, input_dim=784, output_dim=10,
                            DropoutLayer=layers.VariationalDropoutB,
                            n_hidden=100):
    l_in = lasagne.layers.InputLayer((batch_size, input_dim))
    l_hidden_1 = lasagne.layers.DenseLayer(l_in, num_units=n_hidden, 
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_1 = DropoutLayer(l_hidden_1, p=0.2, adaptive="weightwise")
    l_hidden_2 = lasagne.layers.DenseLayer(l_drop_1, num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_2 = DropoutLayer(l_hidden_2, p=0.5, adaptive="weightwise")
    l_out = lasagne.layers.DenseLayer(l_drop_2, num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax)
    l_drop_3 = DropoutLayer(l_out, p=0.5)
    return l_drop_3

def Effect_vardropBDropoutArchitecture(batch_size=1000, input_dim=784, output_dim=10,
                            DropoutLayer=layers.Effect_VariationalDropoutB,
                            n_hidden=100):
    l_in = lasagne.layers.InputLayer((batch_size, input_dim))
    l_hidden_1 = lasagne.layers.DenseLayer(l_in, num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_1 = DropoutLayer(l_hidden_1, p=0.2, adaptive="weightwise")
    l_hidden_2 = lasagne.layers.DenseLayer(l_drop_1, num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_2 = DropoutLayer(l_hidden_2, p=0.5, adaptive="weightwise")
    l_out = lasagne.layers.DenseLayer(l_drop_2, num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax)
    l_drop_3 = DropoutLayer(l_out, p=0.5)
    return l_drop_3

def srivastavaDropoutArchitecture(batch_size=1000, input_dim=784, output_dim=10,
                            DropoutLayer=layers.SrivastavaGaussianDropout,
                            n_hidden=100):
    l_in = lasagne.layers.InputLayer((batch_size, input_dim))
    l_drop_in = DropoutLayer(l_in, p=0.2)
    l_hidden_1 = lasagne.layers.DenseLayer(l_drop_in, num_units=n_hidden, 
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_1 = DropoutLayer(l_hidden_1, p=0.5)
    l_hidden_2 = lasagne.layers.DenseLayer(l_drop_1, num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_2 = DropoutLayer(l_hidden_2, p=0.5)
    l_out = lasagne.layers.DenseLayer(l_drop_2, num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def srivastavaDropoutArchitecture_cifar(batch_size=1000, output_dim=10,
                            DropoutLayer=layers.SrivastavaGaussianDropout,
                            n_hidden=1):
    input_x = T.tensor4("input")
    l_in = lasagne.layers.InputLayer((batch_size, 3, 32, 32), input_x)
    l_drop_in = DropoutLayer(l_in, p=0.5)
    l_hidden_1 = lasagne.layers.Conv2DLayer(l_drop_in, num_filters = int(32*n_hidden),  filter_size=3, stride=(2,2),nonlinearity=lasagne.nonlinearities.softplus)
    #l_hidden_1 = lasagne.layers.MaxPool2DLayer(l_hidden_1, 2)
    #l_hidden_1 = lasagne.nonlinearities.softmax(l_hidden_1)

    l_drop_1 = DropoutLayer(l_hidden_1, p=0.5)
    l_hidden_2 = lasagne.layers.Conv2DLayer(l_drop_1, num_filters = int(64*n_hidden),  filter_size=3, stride=(2,2),nonlinearity=lasagne.nonlinearities.softplus)
    #l_hidden_2 = lasagne.layers.MaxPool2DLayer(l_hidden_2, 2)
    #l_hidden_2 = lasagne.nonlinearities.softmax(l_hidden_2)
    l_drop_2 = DropoutLayer(l_hidden_2, p=0.5)
    # l_hidden_3 = lasagne.layers.DenseLayer(l_drop_2, num_units=int(128 * n_hidden),nonlinearity=lasagne.nonlinearities.rectify)
    l_hidden_3 = lasagne.layers.DenseLayer(l_drop_2, num_units=int(128 * n_hidden), nonlinearity=None)
    l_drop_3 = DropoutLayer(l_hidden_3, p=0.5)
    #l_hidden_4 = lasagne.layers.DenseLayer(l_drop_3, num_units=int(128 * n_hidden), nonlinearity=lasagne.nonlinearities.rectify)
    l_hidden_4 = lasagne.layers.DenseLayer(l_drop_3, num_units=int(128 * n_hidden), nonlinearity=None)
    l_drop_4 = DropoutLayer(l_hidden_4, p=0.5)
    l_out = lasagne.layers.DenseLayer(l_drop_4, num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def vardropADropoutArchitecture(batch_size=1000, input_dim=784, output_dim=10,
                            DropoutLayer=layers.VariationalDropoutA,
                            n_hidden=100):
    l_in = lasagne.layers.InputLayer((batch_size, input_dim))
    l_drop_in = DropoutLayer(l_in, p=0.2, adaptive="elementwise")
    l_hidden_1 = lasagne.layers.DenseLayer(l_drop_in, num_units=n_hidden, 
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_1 = DropoutLayer(l_hidden_1, p=0.5, adaptive="elementwise")
    l_hidden_2 = lasagne.layers.DenseLayer(l_drop_1, num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_2 = DropoutLayer(l_hidden_2, p=0.5, adaptive="elementwise")
    l_out = lasagne.layers.DenseLayer(l_drop_2, num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out



def vardropADropoutArchitecture_cafir(batch_size=100, output_dim=10,
                            DropoutLayer=layers.VariationalDropoutA,
                            n_hidden=1):
    input_x = T.tensor4("input")
    l_in = lasagne.layers.InputLayer((batch_size, 3, 32, 32), input_x)
    l_drop_in = DropoutLayer(l_in, p=0.5, adaptive="elementwise")
    # class lasagne.layers.Conv2DLayer(incoming, num_filters, filter_size, stride=(1, 1), pad=0, untie_biases=False, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d, **kwargs)
    l_hidden_1 = conv_bn_rectify(l_drop_in, 16)
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 16)
    l_hidden_1 = lasagne.layers.MaxPool2DLayer(l_hidden_1, 2, 2)

    # l_hidden_1 = lasagne.layers.MaxPool2DLayer(l_hidden_1, 2)

    l_hidden_1 = DropoutLayer(l_hidden_1, p=0.5, adaptive="elementwise")
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 32)
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 32)
    l_hidden_1 = lasagne.layers.MaxPool2DLayer(l_hidden_1, 2, 2)

    # l_hidden_2 = lasagne.layers.MaxPool2DLayer(l_hidden_2, 2)
    l_hidden_1 = DropoutLayer(l_hidden_1, p=0.5, adaptive="elementwise")
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 64)
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 64)
    l_hidden_1 = lasagne.layers.MaxPool2DLayer(l_hidden_1, 2, 2)

    l_hidden_1 = DropoutLayer(l_hidden_1, p=0.5, adaptive="elementwise")
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 128)
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 128)
    l_hidden_1 = lasagne.layers.MaxPool2DLayer(l_hidden_1, 2, 2)

    l_hidden_1 = DropoutLayer(l_hidden_1, p=0.5, adaptive="elementwise")
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 128)
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 128)
    l_hidden_1 = lasagne.layers.MaxPool2DLayer(l_hidden_1, 2, 2)

    l_hidden_1 = DropoutLayer(l_hidden_1, p=0.5, adaptive="elementwise")
    l_hidden_1 = lasagne.layers.DenseLayer(l_hidden_1, num_units=128,
                                           nonlinearity=None)
    l_hidden_1 = DropoutLayer(l_hidden_1, p=0.5, adaptive="elementwise")
    l_out = lasagne.layers.DenseLayer(l_hidden_1, num_units=output_dim,
                                      nonlinearity=lasagne.nonlinearities.softmax)
    return l_out




def Effect_vardropADropoutArchitecture(batch_size=1000, input_dim=784, output_dim=10,
                            DropoutLayer=layers.Effect_VariationalDropoutA,
                            n_hidden=100):
    l_in = lasagne.layers.InputLayer((batch_size, input_dim))
    l_drop_in = DropoutLayer(l_in, p=0.2, adaptive="elementwise")
    l_hidden_1 = lasagne.layers.DenseLayer(l_drop_in, num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_1 = DropoutLayer(l_hidden_1, p=0.5, adaptive="elementwise")
    l_hidden_2 = lasagne.layers.DenseLayer(l_drop_1, num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_2 = DropoutLayer(l_hidden_2, p=0.5, adaptive="elementwise")
    l_out = lasagne.layers.DenseLayer(l_drop_2, num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out




def Effect_vardropADropoutArchitecture_cifar(batch_size=100, output_dim=10,
                            DropoutLayer=layers.Effect_VariationalDropoutA,
                            n_hidden=100):
    input_x = T.tensor4("input")
    l_in = lasagne.layers.InputLayer((batch_size, 3, 32, 32), input_x)
    l_drop_in = DropoutLayer(l_in, p=0.5, adaptive="elementwise")
    #class lasagne.layers.Conv2DLayer(incoming, num_filters, filter_size, stride=(1, 1), pad=0, untie_biases=False, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d, **kwargs)
    l_hidden_1 = conv_bn_rectify(l_drop_in, 16)
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 16)
    l_hidden_1 = lasagne.layers.MaxPool2DLayer(l_hidden_1,2,2)


    #l_hidden_1 = lasagne.layers.MaxPool2DLayer(l_hidden_1, 2)

    l_hidden_1 = DropoutLayer(l_hidden_1, p=0.5, adaptive="elementwise")
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 32)
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 32)
    l_hidden_1 = lasagne.layers.MaxPool2DLayer(l_hidden_1, 2, 2)


    #l_hidden_2 = lasagne.layers.MaxPool2DLayer(l_hidden_2, 2)
    l_hidden_1 = DropoutLayer(l_hidden_1, p=0.5, adaptive="elementwise")
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 64)
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 64)
    l_hidden_1 = lasagne.layers.MaxPool2DLayer(l_hidden_1, 2, 2)

    l_hidden_1 = DropoutLayer(l_hidden_1, p=0.5, adaptive="elementwise")
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 128)
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 128)
    l_hidden_1 = lasagne.layers.MaxPool2DLayer(l_hidden_1, 2, 2)

    l_hidden_1 = DropoutLayer(l_hidden_1, p=0.5, adaptive="elementwise")
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 128)
    l_hidden_1 = conv_bn_rectify(l_hidden_1, 128)
    l_hidden_1 = lasagne.layers.MaxPool2DLayer(l_hidden_1, 2, 2)

    l_hidden_1 = DropoutLayer(l_hidden_1, p=0.5, adaptive="elementwise")
    l_hidden_1 = lasagne.layers.DenseLayer(l_hidden_1, num_units=128,
            nonlinearity=None)
    l_hidden_1 = DropoutLayer(l_hidden_1, p=0.5, adaptive="elementwise")
    l_out = lasagne.layers.DenseLayer(l_hidden_1, num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out
def sparsevardropDropoutArchitecture(batch_size=1000, input_dim=784, output_dim=10,
                            DropoutLayer=layers.SparsityVariationalDropout,
                            n_hidden=100):
    l_in = lasagne.layers.InputLayer((batch_size, input_dim))
    l_drop_in = DropoutLayer(l_in, p=0.2, adaptive="elementwise")
    l_hidden_1 = lasagne.layers.DenseLayer(l_drop_in, num_units=n_hidden, 
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_1 = DropoutLayer(l_hidden_1, p=0.5, adaptive="elementwise")
    l_hidden_2 = lasagne.layers.DenseLayer(l_drop_1, num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_drop_2 = DropoutLayer(l_hidden_2, p=0.5, adaptive="elementwise")
    l_out = lasagne.layers.DenseLayer(l_drop_2, num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def make_experiment(l_out, dataset, batch_size=1000,
        N_train=50000, N_valid=10000, N_test=10000,
        loss_function=lasagne.objectives.categorical_crossentropy,
        extra_loss=0.0):
    """
    Build a loop for training a model, evaluating loss on training, validation 
    and test.
    """
    expressions = holonets.monitor.Expressions(l_out, dataset, 
            batch_size=batch_size, update_rule=lasagne.updates.adam, 
            loss_function=loss_function, loss_aggregate=T.mean, 
            extra_loss=extra_loss, learning_rate=0.001, momentum=0.1)
    # only add channels for loss and accuracy
    for deterministic,dataset in zip([False, True, True],
                                     ["train", "valid", "test"]):
        expressions.add_channel(**expressions.loss(dataset, deterministic))
        expressions.add_channel(**expressions.accuracy(dataset, deterministic))
    channels = expressions.build_channels()
    train = holonets.train.Train(channels, 
            n_batches={'train': N_train//batch_size, 
                       'valid':N_valid//batch_size, 
                       'test':N_test//batch_size})
    loop = holonets.run.EpochLoop(train, dimensions=train.dimensions)
    return loop


def make_experiment_cifar(l_out, dataset, batch_size=1000,
        N_train=50000, N_valid=0, N_test=10000,
        loss_function=lasagne.objectives.categorical_crossentropy,
        extra_loss=0.0):
    """
    Build a loop for training a model, evaluating loss on training, validation
    and test.
    """
    expressions = holonets.monitor.Expressions_cifar10(l_out, dataset,
            batch_size=batch_size, update_rule=lasagne.updates.adam,
            loss_function=loss_function, loss_aggregate=T.mean,
            extra_loss=extra_loss, learning_rate=0.001, momentum=0.1)
    # only add channels for loss and accuracy
    for deterministic,dataset in zip([False, True],
                                     ["train", "test"]):
        expressions.add_channel(**expressions.loss(dataset, deterministic))
        expressions.add_channel(**expressions.accuracy(dataset, deterministic))
    channels = expressions.build_channels()
    train = holonets.train.Train(channels,
            n_batches={'train': N_train//batch_size,
                       # 'valid':N_valid//batch_size,
                       'test':N_test//batch_size})
    loop = holonets.run.EpochLoop(train, dimensions=train.dimensions)
    return loop
def make_experiment_svhn(l_out, dataset, batch_size=1000,
        N_train=73257, N_valid=0, N_test=26032,
        loss_function=lasagne.objectives.categorical_crossentropy,
        extra_loss=0.0):
    """
    Build a loop for training a model, evaluating loss on training, validation
    and test.
    """
    expressions = holonets.monitor.Expressions_cifar10(l_out, dataset,
            batch_size=batch_size, update_rule=lasagne.updates.adam,
            loss_function=loss_function, loss_aggregate=T.mean,
            extra_loss=extra_loss, learning_rate=0.001, momentum=0.1)
    # only add channels for loss and accuracy
    for deterministic,dataset in zip([False, True],
                                     ["train", "test"]):
        expressions.add_channel(**expressions.loss(dataset, deterministic))
        expressions.add_channel(**expressions.accuracy(dataset, deterministic))
    channels = expressions.build_channels()
    train = holonets.train.Train(channels,
            n_batches={'train': N_train//batch_size,
                       # 'valid':N_valid//batch_size,
                       'test':N_test//batch_size})
    loop = holonets.run.EpochLoop(train, dimensions=train.dimensions)
    return loop
def make_experiment_cifar1(l_out, dataset, batch_size=1000,
        N_train=40000, N_valid=10000, N_test=10000,
        loss_function=lasagne.objectives.categorical_crossentropy,
        extra_loss=0.0):
    """
    Build a loop for training a model, evaluating loss on training, validation
    and test.
    """
    expressions = holonets.monitor.Expressions_cifar101(l_out, dataset,
            batch_size=batch_size, update_rule=lasagne.updates.adam,
            loss_function=loss_function, loss_aggregate=T.mean,
            extra_loss=extra_loss, learning_rate=0.001, momentum=0.1)
    # only add channels for loss and accuracy
    for deterministic,dataset in zip([False, True, True],
                                     ["train", "valid", "test"]):
        expressions.add_channel(**expressions.loss(dataset, deterministic))
        expressions.add_channel(**expressions.accuracy(dataset, deterministic))
    channels = expressions.build_channels()
    train = holonets.train.Train(channels,
            n_batches={'train': N_train//batch_size,
                       'valid': N_valid//batch_size,
                       'test': N_test//batch_size})
    loop = holonets.run.EpochLoop(train, dimensions=train.dimensions)
    return loop



def earlystopping(loop, delta=0.001, max_N=50, verbose=False, lookback=1):
    """
    Stops the expriment once the loss stops improving by delta per epoch.
    With a max_N of epochs to avoid infinite experiments.
    """
    prev_loss, loss_diff = 100, 0.9
    N = 0
    #while -loss_diff < delta and N < max_N:
    while N < max_N:
        # run one epoch
        results = loop.run(1)
        N += 1
        current_loss = loop.results["valid Loss"][-1][1]
        loss_diff = (prev_loss-current_loss)/prev_loss
        if verbose:
            print(N, loss_diff)
        prev_loss = loop.results["valid Loss"][-min([lookback,
                                len(loop.results['valid Loss'])])][1]
    return results

def earlystopping_cifar(loop, delta=0.001, max_N=100, verbose=False, lookback=1):
    """
    Stops the expriment once the loss stops improving by delta per epoch.
    With a max_N of epochs to avoid infinite experiments.
    """
    prev_loss, loss_diff = 100, 0.9
    N = 0
    #while -loss_diff < delta and N < max_N:
    while N < max_N:
        # run one epoch
        results = loop.run(1)
        N += 1
        current_loss = loop.results["test Loss"][-1][1]
        loss_diff = (prev_loss-current_loss)/prev_loss
        if verbose:
            print(N, loss_diff)
        prev_loss = loop.results["test Loss"][-min([lookback,
                                len(loop.results['test Loss'])])][1]
    return results


def earlystopping_cifar1(loop, delta=0.001, max_N=200, verbose=False, lookback=1):
    """
    Stops the expriment once the loss stops improving by delta per epoch.
    With a max_N of epochs to avoid infinite experiments.
    """
    prev_loss, loss_diff = 100, 0.9
    N = 0
    #while -loss_diff < delta and N < max_N:
    while N < max_N:
        # run one epoch
        results = loop.run(1)
        N += 1
        current_loss = loop.results["valid Loss"][-1][1]
        loss_diff = (prev_loss-current_loss)/prev_loss
        if verbose:
            print(N, loss_diff)
        prev_loss = loop.results["valid Loss"][-min([lookback,
                                len(loop.results['valid Loss'])])][1]
    return results

def load_data():
    """
    Standardising data loading; all using MNIST in the usual way:
        * train: 50000
        * valid: 10000
        * test: separate 10000
    """
    # is this the laziest way to load mnist?
    mnist = imp.new_module('mnist')
    # datau = urllib.request.urlopen("https://raw.githubusercontent.com/Lasagne/Lasagne" "/master/examples/mnist.py")
    # dataset = datau.read()
    datau = urllib.request.urlopen("https://raw.githubusercontent.com/Lasagne/Lasagne"
            "/master/examples/mnist.py")
    (mnist.read() in mnist.__dict__)
    dataset = mnist.load_dataset()
    return dict(X_train=dataset[0].reshape(-1, 784),
                y_train=dataset[1],
                X_valid=dataset[2].reshape(-1, 784),
                y_valid=dataset[3],
                X_test=dataset[4].reshape(-1, 784),
                y_test=dataset[5])



def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    # return X_train, y_train, X_val, y_val, X_test, y_test
    return dict(X_train=X_train.reshape(-1, 784),
                y_train=y_train,
                X_valid=X_val.reshape(-1, 784),
                y_valid=y_val,
                X_test=X_test.reshape(-1, 784),
                y_test=y_test)



def load_cifar10_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    # return X_train, y_train, X_val, y_val, X_test, y_test
    return dict(X_train=X_train.reshape(-1, 784),
                y_train=y_train,
                X_valid=X_val.reshape(-1, 784),
                y_valid=y_val,
                X_test=X_test.reshape(-1, 784),
                y_test=y_test)




def load_cifar10( ):
    def load_CIFAR_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f , encoding='latin1')
            Y = np.array(datadict['labels'])
            X = datadict['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            X = X/255.0
            Y_oh = np.zeros([len(Y), 10])
            Y_oh[np.arange(len(Y)), Y] = 1
            return X, Y# Y_oh

    def load_CIFAR10( ):
        xs, ys = [], []
        for b in range(1, 6):
            X, Y = load_CIFAR_batch(('D:/PyCharm/2018Works/sbp/data/cifar10/cifar-10-batches-py/data_batch_%d' % (b,)))
            xs.append(X)
            ys.append(Y)
        Xtr, Ytr = np.concatenate(xs), np.concatenate(ys)
        del X, Y
        Xte, Yte = load_CIFAR_batch(('D:/PyCharm/2018Works/sbp/data/cifar10/cifar-10-batches-py/' + 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = load_CIFAR10( )

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    #return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 32, 32, 3), 10
    return dict(X_train=X_train,
                y_train=y_train,
                #X_valid=X_val.reshape(-1, 784),
                #y_valid=y_val,
                X_test=X_test,
                y_test=y_test)




def load_cifar101( ):
    def load_CIFAR_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f , encoding='latin1')
            Y = np.array(datadict['labels'])
            X = datadict['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            X = X/255.0
            Y_oh = np.zeros([len(Y), 10])
            Y_oh[np.arange(len(Y)), Y] = 1
            return X, Y# Y_oh

    def load_CIFAR10( ):
        xs, ys = [], []
        for b in range(1, 6):
            X, Y = load_CIFAR_batch(('D:/PyCharm/2018work/group-sparsity-sbp-master/data/cifar10/cifar-10-batches-py/data_batch_%d' % (b,)))
            xs.append(X)
            ys.append(Y)
        Xtr, Ytr = np.concatenate(xs), np.concatenate(ys)
        del X, Y
        Xte, Yte = load_CIFAR_batch(('D:/PyCharm/2018work/group-sparsity-sbp-master/data/cifar10/cifar-10-batches-py/' + 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = load_CIFAR10( )

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_valid = X_train[40000:,:,:,:]
    X_train = X_train[:40000,:,:,:]
    y_valid = y_train[40000:]
    y_train = y_train[:40000]

    X_test = X_test.transpose(0, 3, 1, 2).copy()

    #return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 32, 32, 3), 10
    return dict(X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                y_test=y_test)

def get_argparser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--output_directory", "-results", help="directory to save pickle ")
    parser.add_argument("-output_directory", help="directory to save pickle "
                                                 "files of results", default="results")
    parser.add_argument("-v", action='store_true', 
            help="make the experiment more verbose", default=True)
    return parser
