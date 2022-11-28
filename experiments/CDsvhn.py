from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import layers as tfl

from tensorflow.examples.tutorials.mnist import input_data

from concrete_dropout import concrete_dropout

import math


from cifar10.include.data import get_data_set
from cifar10.include.model import model, lr




import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy.io import loadmat
#from skimage import color
#from skimage import io
#from sklearn.model_selection import train_test_split



def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    for i in range(len(data['y'])):
        if data['y'][i]==10:
            data['y'][i] =0
    return data['X']/255.0, data['y']


_BATCH_SIZE = 10
def net(inputs, is_training,n_hidden=100):

    #x = tf.reshape(inputs, [-1, 28*28])
    x = tf.reshape(inputs, [-1, 32, 32, 3])

    dropout_params = {'init_min': 0.9, 'init_max': 0.9,
                      'weight_regularizer': 1e-6, 'dropout_regularizer': 1e-5,
                      'training': is_training}
    dropout_params1 = {'init_min': 0.75, 'init_max': 0.75,
                      'weight_regularizer': 1e-6, 'dropout_regularizer': 1e-5,
                      'training': is_training}
    dropout_params2 = {'init_min': 0.5, 'init_max': 0.5,
                      'weight_regularizer': 1e-6, 'dropout_regularizer': 1e-5,
                      'training': is_training}

    x, reg = concrete_dropout(x, name='conv1_dropout', **dropout_params)
    x = tfl.conv2d(x, 92, 5, strides=(1,1), activation=tf.nn.relu, padding='SAME',
                   kernel_regularizer=reg, bias_regularizer=reg,
                   name='conv1')
    x = tfl.max_pooling2d(x, pool_size=(3,3),strides=(2,2))

    x, reg = concrete_dropout(x, name='conv2_dropout', **dropout_params1)
    x = tfl.conv2d(x, 128, 5, strides=(1,1), activation=tf.nn.relu, padding='SAME',
                   kernel_regularizer=reg, bias_regularizer=reg,
                   name='conv2')
    x = tfl.max_pooling2d(x, pool_size=(3,3),strides=(2,2))


    x, reg = concrete_dropout(x, name='conv3_dropout', **dropout_params1)
    x = tfl.conv2d(x, 256, 5, strides=(1, 1), activation=tf.nn.relu, padding='SAME',
                   kernel_regularizer=reg, bias_regularizer=reg,
                   name='conv3')
    x = tfl.max_pooling2d(x, pool_size=(3,3),strides=(2,2))


    x = tf.reshape(x, [-1, 3 * 3 * 256], name='flatten')
    x, reg = concrete_dropout(x, name='fc1_dropout', **dropout_params2)
    x = tfl.dense(x, 2048, activation=tf.nn.relu, name='fc1',
                  kernel_regularizer=reg, bias_regularizer=reg)
    x, reg = concrete_dropout(x, name='fc2_dropout', **dropout_params2)
    x = tfl.dense(x, 2048, activation=tf.nn.relu, name='fc2',
                  kernel_regularizer=reg, bias_regularizer=reg)
    x, reg = concrete_dropout(x, name='fc3_dropout', **dropout_params2)

    outputs = tfl.dense(x, 10, name='fc3')
    return outputs


def main(_):
    #mnist = input_data.read_data_sets('MNIST_data',validation_size=10000)

    #train_x, train_y = get_data_set("train")
    #test_x, test_y = get_data_set("test")

    train_x, train_y = load_data('train_32x32.mat')
    test_x, test_y = load_data('test_32x32.mat')
    train_x, train_y = train_x.transpose((3, 0, 1, 2)), train_y[:, 0]
    test_x, test_y = test_x.transpose((3, 0, 1, 2)), test_y[:, 0]


    x = tf.placeholder(tf.float32, [None, 32,32,3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)
    n_hidden=100

    y_out = net(x, is_training,n_hidden)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
                labels=y, logits=y_out))
        loss += tf.reduce_sum(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_out, 1), y)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    dropout_rates = tf.get_collection('DROPOUT_RATES')
    def rates_pretty_print(values):
        return {str(t.name): round(r, 4)
                for t, r in zip(dropout_rates, values)}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
        prev_loss, loss_diff = 100,0.9
        counter = 0
        while counter<100:
            counter = counter +1
            for i in range(batch_size):
                #batch = mnist.train.next_batch(1000)
                batch_xs = train_x[i * _BATCH_SIZE: (i + 1) * _BATCH_SIZE]
                batch_ys = train_y[i * _BATCH_SIZE: (i + 1) * _BATCH_SIZE]
                if i % 499 == 0:
                    training_loss, training_acc, rates = sess.run(
                        [loss, accuracy, dropout_rates],
                        feed_dict={
                            x: batch_xs, y: batch_ys, is_training: False})
                    print('step {}, loss {}, accuracy {}'.format(
                        counter, training_loss, training_acc))
                    print('dropout rates: {}'.format(rates_pretty_print(rates)))



                    current_loss, accuracy1, rates1 = sess.run([loss, accuracy, dropout_rates],
                                               feed_dict={x: test_x,
                                                          y: test_y,
                                                          is_training: False})
                    print('test accuracy {}'.format(accuracy1))
                    loss_diff = (prev_loss - current_loss) / prev_loss
                    prev_loss = current_loss
                    print('loss_diff', loss_diff)

                train_step.run(feed_dict={
                    x: batch_xs, y: batch_ys, is_training: True})





        accuracy, rates = sess.run([accuracy, dropout_rates],
                                   feed_dict={x: test_x,
                                              y: test_y,
                                              is_training: False})
        print('test accuracy {}'.format(accuracy))
        print('final dropout rates: {}'.format(rates_pretty_print(rates)))


if __name__ == '__main__':
    tf.app.run(main=main)