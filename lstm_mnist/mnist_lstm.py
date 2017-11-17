import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data", one_hot=True)
print("data shape:", mnist.train.images.shape)
print("data shape:", mnist.test.images.shape)
print("data shape:", mnist.test.labels.shape)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

from tensorflow.contrib import rnn

lstm_cell = rnn.BasicLSTMCell(num_units = hidden_size, forget_bias=1, state_is_tuple = True )
mlstm_cell = rnn.MultiRNNCell([lstm_cell] * 2, state_is_tuple = True)
init_state = mlstm_cell.zero_state(batch_size, dtype = tf.float32)

output = []
state = init_state
with tf.variable_scope("lstm"):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        cell_output, state = mlstm_cell(X[:, timestep, :], state)
        output.append( cell_output)
h_state = output[-1]

W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev = 0.1), dtype = tf.float32)
b = tf.Variable(tf.constant(0, shape=[class_num]), dtype = tf.float32)
_Y = tf.nn.softmax(tf.matmul(h_state, W)+b)

ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = _Y, labels = Y))
train_op = tf.train.AdamOptimizer(lr).minimize(ce)

correct_prediction - tf.equal(tf.argmax(_Y, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as ss:
    ss.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(batch_size)
        ss.run(train_op, feed_dict = {_X:batch[0], Y:batch[1] })
        if ((i+1)%200) == 0:
            train_accurary = ss.run( accuracy, feed_dict = {_X:mnist.test.images, Y: mnist.test.labels })
            print("acc" , train_accurary )





