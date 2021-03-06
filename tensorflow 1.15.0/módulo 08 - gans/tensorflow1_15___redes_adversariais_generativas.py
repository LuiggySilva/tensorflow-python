# -*- coding: utf-8 -*-
"""tensorflow1_15___redes_adversariais_generativas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1s0S2O-DGXybeHDW0whMMianHcZfq-a9K
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.reset_default_graph()

"""> # **Aula 023** - Redes Adversariais Generativas - **Tensorflow: GANs**"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist/', one_hot= True)
plt.imshow(mnist.train.images[5].reshape(28,28), cmap='Greys')

img_01 = np.arange(0, 784).reshape(28,28)
plt.imshow(img_01)

img_02 = np.random.normal(size= 784).reshape(28,28)
plt.imshow(img_02)

ruido_ph = tf.placeholder(tf.float32, shape= [None, 100])

def gerador(ruido, _reuse= True):
  with tf.variable_scope('gerador', reuse= _reuse):
    # Estrutura da rede
    # entrada 100 -> oculta1 128 -> oculta2 128 -> saida 784
    camada_oculta1 = tf.nn.relu( tf.layers.dense(inputs= ruido, units= 128) )
    camada_oculta2 = tf.nn.relu( tf.layers.dense(inputs= camada_oculta1, units= 128) )
    camada_saida = tf.layers.dense(inputs= camada_oculta2, units= 784, activation= tf.nn.tanh)
    return camada_saida

img_reais_ph = tf.placeholder(tf.float32, [None, 784])

def discriminador(X, _reuse= True):
  with tf.variable_scope('discriminador', reuse= _reuse):
    # Estrutura da rede
    # entrada 784 -> oculta1 128 -> oculta2 128 -> saida 1
    camada_oculta1 = tf.nn.relu( tf.layers.dense(inputs= X, units= 128) )
    camada_oculta2 = tf.nn.relu( tf.layers.dense(inputs= camada_oculta1, units= 128) )
    logits = tf.layers.dense(inputs= camada_oculta2, units= 1)
    return logits

logits_img_reais = discriminador(img_reais_ph, None)
logits_img_ruidos = discriminador(gerador(ruido_ph, None), True)

erro_discriminador_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits= logits_img_reais, labels= tf.ones_like(logits_img_reais) * 0.9)
)
erro_discriminador_ruido = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits= logits_img_ruidos, labels= tf.zeros_like(logits_img_reais))
)
erro_discriminador = erro_discriminador_real + erro_discriminador_ruido

erro_gerador = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits= logits_img_ruidos, labels= tf.ones_like(logits_img_ruidos))
)

variaveis = tf.trainable_variables()
variaveis

variaveis_discriminador = [v for v in variaveis if 'discriminador' in v.name]
variaveis_discriminador

variaveis_gerador = [v for v in variaveis if 'gerador' in v.name]
variaveis_gerador

treinamento_discriminador = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(erro_discriminador,
                                                                                  var_list= variaveis_discriminador)
treinamento_gerador = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(erro_gerador,
                                                                            var_list= variaveis_gerador)

batch_size = 100
amostras_teste = []
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  #(-1, 1, ... ) -> tf.nn.tanh | (0, 1, ...) tf.nn.sigmoid
  #ruido_testes = np.random.uniform(-1, 1, size=(1, 100))
  #amostra = sess.run(generador(ruido_ph), feed_dict= {ruido_ph: ruido_testes})

  for epoch in range(500):
    num_batches = mnist.train.num_examples // batch_size
    for i in range(num_batches):
      batch = mnist.train.next_batch(batch_size)
      img_batch = batch[0].reshape((100,784))
      img_batch = img_batch * 2 - 1
      
      batch_ruido = np.random.uniform(-1, 1, size=(batch_size, 100))
      _, custod = sess.run([treinamento_discriminador, erro_discriminador], 
                          feed_dict= {img_reais_ph: img_batch, ruido_ph: batch_ruido})
      _, custog = sess.run([treinamento_gerador, erro_gerador], 
                           feed_dict= {ruido_ph: batch_ruido})
      
    print(f'epoch {epoch} - erroD {custod} - erroG {custog}')
    
    ruido_teste = np.random.uniform(-1, 1, size=(1, 100))
    img_gerada = sess.run(gerador(ruido_ph), feed_dict= {ruido_ph: ruido_teste})
    amostras_teste.append(img_gerada)

plt.imshow(amostras_teste[0].reshape(28,28))

plt.imshow(amostras_teste[99].reshape(28,28))

plt.imshow(amostras_teste[199].reshape(28,28))

plt.imshow(amostras_teste[299].reshape(28,28))

plt.imshow(amostras_teste[399].reshape(28,28))

plt.imshow(amostras_teste[499].reshape(28,28))