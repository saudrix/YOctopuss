
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


def new_layer(a1, sz):
    with tf.name_scope('couche'):
        w = tf.Variable(tf.random_normal([sz[0], sz[1]]), name = 'poids')
        b = tf.Variable(tf.random_normal([sz[1]]), name = 'biais')
        a = tf.nn.relu(tf.add(tf.matmul(a1, w), b))
        tf.summary.histogram('poids', w)
        tf.summary.histogram('biais', b)
        tf.summary.histogram('activation', a)

    return a


# In[ ]:


def MLP(nb_couches, nb_neurons, h, p):
    """
        Permet de créer un perceptron multicouches, paramétrable en fonction de:
            - nb_couches : le nombre de couches cachées
            - nb_neurones : une liste qui contient le nombre de neurones sur chaque couches
            - h : le learning_rate
            - p : un identifiant quelconque (utile pour l'affiche sous tensorboard)
    """
    tf.reset_default_graph()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape = [None, 784])
        y_ = tf.placeholder(tf.float32, shape = [None, 10])

    with tf.name_scope('couche'):
        w = tf.Variable(tf.random_normal([784, nb_neurons[0]]), name = 'poids')
        b = tf.Variable(tf.random_normal([nb_neurons[0]]), name = 'biais')
        activations = [tf.nn.relu(tf.add(tf.matmul(x, w), b))]

    for n in range(nb_couches-1):
        activations.append(new_layer(activations[-1], [nb_neurons[n], nb_neurons[n+1]]))

    with tf.name_scope('Output'):
        w_out = tf.Variable(tf.random_normal([nb_neurons[-1], 10]), name = 'poids')
        b_out = tf.Variable(tf.random_normal([10]), name = 'biais')
        y = tf.add(tf.matmul(activations[-1], w_out), b_out, name = 'y')

    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))

    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train = tf.train.AdamOptimizer(learning_rate=h).minimize(cost)


    with tf.name_scope('summaries'):
        tf.summary.scalar('cout', cost)
        tf.summary.scalar('precision', accuracy)
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 3)

    merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    model = tf.global_variables_initializer()

    train_writer = tf.summary.FileWriter('output/MLP_{};eta={};{}/train'.format(nb_couches, h, p))
    test_writer = tf.summary.FileWriter('output/MLP_{};eta={};{}/test'.format(nb_couches, h, p))

    steps = 20
    batch = 300
    mnist_size = int(mnist.train.num_examples)

    # On créer une session pour computer le graphe
    with tf.Session() as session:
        session.run(model)
        train_writer.add_graph(session.graph)

        s = -1
        for i in range(steps):
            moyenne = 0
            epoch_loss  = 0

            for j in range(mnist_size//batch):
                s += 1
                x_train, y_train = mnist.train.next_batch(batch)
                _, c, summary_t = session.run([train, cost, merged], feed_dict = {x: x_train, y_: y_train})
                epoch_loss += c

                if s%100 == 0:
                    x_test, y_test = mnist.test.images, mnist.test.labels
                    acc, c, summary = session.run([accuracy,cost, merged], feed_dict = {x: x_test, y_: y_test})
                    train_writer.add_summary(summary_t, s)
                    test_writer.add_summary(summary, s)
                    print('Step : {} | Accuracy : {} | loss : {}'.format(s, acc, c))
                    saver.save(session, "tmp/model_{}_{}_.ckpt".format(nb_couches,p), s)

    #print(acc)
    return acc


# In[ ]:


# On définit une liste de réseaux
l = [[7, [756, 551, 638, 773, 350, 604, 730], 0.009]]

p = 1
# On entraine tous les réseaux
for par in l:
    p+=1
    MLP(*par, p)
