
# coding: utf-8

# In[ ]:


# import some stuff
import matplotlib.pyplot as plt
import numpy as np
from time import time
from random import randrange

# import tensorflow
import tensorflow as tf

# import MNIST database
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[ ]:


def conv_layer(a, inner, out):
    w = tf.Variable(tf.random_normal([5,5,inner,out]))
    b = tf.Variable(tf.random_normal([out]))

    with tf.name_scope('Conv_Layer'):
        conv = tf.nn.relu(tf.nn.conv2d(a, w, strides=[1,1,1,1], padding='SAME') + b)
        pool = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



    return pool


# In[ ]:


def fully_connected_layer(a, inner, out, keep_rate):
    with tf.name_scope('Fully_Connected'):
        w = tf.Variable(tf.random_normal([inner,out]))
        b = tf.Variable(tf.random_normal([out]))
        fc = tf.nn.relu(tf.matmul(a, w) + b)
        drop = tf.nn.dropout(fc, keep_rate)



    return drop


# In[ ]:


def CNN():
    tf.reset_default_graph()
    steps = 20
    batch_size = 1
    learning_rate = 0.001
    keep_rate = 0.8
    keep_prob = tf.placeholder(tf.float32)
    with tf.device('/cpu:0'):
        with tf.name_scope('input'):
            x1 = tf.placeholder(tf.float32, [None, 784])
            y = tf.placeholder(tf.float32)

        x = tf.reshape(x1, shape=[-1, 28, 28, 1])
        tf.summary.image('input', x, 3)

    with tf.device('/gpu:0'):
        c1 = conv_layer(x, 1, 32)
        c2 = conv_layer(c1, 32, 64)

    with tf.device('/cpu:0'):
        # 256 is size of last conv_layer, 2*2 is size of image after 4 pooling
        fc1 = tf.reshape(c2,[-1, 2*2*64])
        fc1 = fully_connected_layer(fc1, 2*2*64, 1024, keep_rate)
        fc2 = fully_connected_layer(fc1, 1024, 512, keep_rate)

        with tf.name_scope('output'):
            w = tf.Variable(tf.random_normal([512,10]))
            b = tf.Variable(tf.random_normal([10]))
            output = tf.matmul(fc2, w) + b

    # Cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = output, labels = y))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    with tf.name_scope('summaries'):
        tf.summary.scalar('precision', accuracy)
        tf.summary.scalar('cout', cost)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('output/train/deep_cnn4')
        test_writer = tf.summary.FileWriter('output/test/deep_cnn4')

    # To save model
    saver = tf.train.Saver()

    model = tf.global_variables_initializer()

    # Launch session
    with tf.Session() as sess:
        sess.run(model)

        # Add graph to tensorboard
        train_writer.add_graph(sess.graph)

        s = -1
        for epoch in range(steps):
            epoch_loss = 0
            for k in range(int(mnist.train.num_examples/batch_size)):
                s += 1

                # Get next batch in the MNIST
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                # Train step
                _, c, ac, summary = sess.run([optimizer, cost, accuracy, merged], feed_dict={x1: epoch_x, y: epoch_y})
                epoch_loss += c

                # Add in summary
                train_writer.add_summary(summary, s)

                if s%100 == 0:
                    # Each 100 steps, test step
                    acc, c, summary_t = sess.run([accuracy, cost, merged], feed_dict={x1:mnist.test.images, y:mnist.test.labels})
                    if s > 2000 and acc < 0.5:
                        return

                    test_writer.add_summary(summary_t, s)
                    print('Accuracy:', acc)
                    saver.save(sess, "tmp/model_deep_cnn4_.ckpt", s)

            print('Epoch', epoch, 'completed out of', steps, 'loss:', epoch_loss)




# In[ ]:


CNN()
