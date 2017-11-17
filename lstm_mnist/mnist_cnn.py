import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data", one_hot=True)
print("data shape:", mnist.train.images.shape)
print("data shape:", mnist.test.images.shape)
print("data shape:", mnist.test.labels.shape)



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant( 0.0,dtype=tf.float32, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,W, strides=[1]*4, padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def relu_conv2d(W, b, x):
    return tf.nn.relu(conv2d(x, W)+b)

def drop_out(x):
    keep_prob = tf.placeholder(tf.float32)
    return tf.nn.dropout(x, keep_prob), keep_prob

def linear(W,b,x):
    return tf.matmul(x, W) + b

def relu_linaer(W,b,x):
    return tf.nn.relu(linear(W,b,x))

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
x_iamge = tf.reshape(X, [-1,28,28, 1])
h_conv1 = relu_conv2d(W_conv1, b_conv1, x_iamge)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = relu_conv2d(W_conv2, b_conv2, h_pool1)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = relu_linaer(W_fc1, b_fc1, h_pool2_flat)

h_fc1_dropout, keep_prob = drop_out(h_fc1)
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])


_Y = linear(W_fc2, b_fc2, h_fc1_dropout)

ce = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = _Y, labels = Y))
correct = tf.equal(tf.argmax(Y,1), tf.argmax(_Y,1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))
train_op = tf.train.AdamOptimizer(0.001).minimize(ce)

#exit()
with tf.Session() as ss:
    ss.run(tf.global_variables_initializer())
    for step in range(30000):
        batch = mnist.train.next_batch(100)
        acc_v , _ = ss.run([ acc, train_op], feed_dict = {X:batch[0], Y:batch[1], keep_prob:0.5 })
        #print("train acc:", acc_v)
        if ((step+1) % 10) == 0:
            acc_v = ss.run(acc, feed_dict = {X:mnist.test.images,Y:mnist.test.labels, keep_prob:1.0})
            print("==> test acc:",acc_v)

    

