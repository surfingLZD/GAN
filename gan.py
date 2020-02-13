import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

mnist=input_data.read_data_sets("/Users/mac/Documents/dataSet")

batch_size=32
real=tf.placeholder(tf.float32,shape=[None,784])
gen=tf.placeholder(tf.float32,shape=[None,100])

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


weights={
    "D_h1":tf.Variable(tf.random_normal([784,128])),
    "D_h2":tf.Variable(tf.random_normal([128,1])),
    "G_h1":tf.Variable(tf.random_normal([100,128])),
    "G_h2":tf.Variable(tf.random_normal([128,784]))
}

biases={
    "D_b1":tf.Variable(tf.random_normal([128])),
    "D_b2":tf.Variable(tf.random_normal([1])),
    "G_b1":tf.Variable(tf.random_normal([128])),
    "G_b2":tf.Variable(tf.random_normal([784]))
}

def discriminator(x):
    layer_1=tf.nn.relu(tf.add(tf.matmul(x,weights['D_h1']),biases['D_b1']))
    return tf.add(tf.matmul(layer_1,weights['D_h2']),biases['D_b2'])

def generator(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['G_h1']), biases['G_b1']))
    return tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['G_h2']), biases['G_b2']))


G_sample=generator(gen)
X_real=discriminator(real)
X_gen=discriminator(G_sample)

import os
if not os.path.exists('/Users/mac/Documents/dataSet'):
    os.makedirs('/Users/mac/Documents/dataSet')


D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=X_real, labels=tf.ones_like(X_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=X_gen, labels=tf.zeros_like(X_gen)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=X_gen, labels=tf.ones_like(X_gen)))

#D_loss = -(tf.reduce_mean(D_real) - tf.reduce_mean(0.25*D_fake**2 + D_fake))
#G_loss = -tf.reduce_mean(0.25*D_fake**2 + D_fake)


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


G_optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(G_loss,
                                                                var_list=[weights['G_h1'],weights['G_h2'],biases['G_b1'],biases['G_b2']])
D_optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(D_loss,
                                                                var_list=[weights['D_h1'],weights['D_h2'],biases['D_b1'],biases['D_b2']])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(40):
        for j in range(mnist.train.num_examples//batch_size):
            real_,_=mnist.train.next_batch(batch_size)
            gen_=np.random.uniform(-1,1,size=[batch_size,100])
            _,Gloss=sess.run([G_optimizer,G_loss],feed_dict={gen:gen_})
            _,Dloss=sess.run([D_optimizer,D_loss],feed_dict={real:real_,gen:gen_})
        print(Gloss,Dloss)

        samples = sess.run(G_sample, feed_dict={gen:np.random.uniform(-1,1,size=[16,100])})
        fig = plot(samples)
        plt.savefig('/Users/mac/Documents/dataSet/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)


