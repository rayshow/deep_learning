import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data", one_hot=True)
print("data shape:", mnist.train.images.shape)
print("data shape:", mnist.test.images.shape)
print("data shape:", mnist.test.labels.shape)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
_Y = tf.matmul(X, W) + b

ce = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = _Y, labels = Y))
correct = tf.equal(tf.argmax(Y,1), tf.argmax(_Y,1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(ce)

#exit()
with tf.Session() as ss:
    ss.run(tf.global_variables_initializer())
    for step in range(300000):
        batch = mnist.train.next_batch(100)
        acc_v , _ = ss.run([ acc, train_op], feed_dict = {X:batch[0], Y:batch[1] })
        #print("train acc:", acc_v)
        if ((step+1) % 1000) == 0:
            acc_v = ss.run(acc, feed_dict = {X:mnist.test.images,Y:mnist.test.labels})
            print("==> test acc:",acc_v)

    

