# -*- coding: utf-8 -*-
"""tensorflow2.1.0___treinamento distribuido.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1faaUATpQDqTwBScIQ0s2WKUNTTkkg2zM

> # **TensorFlow  2.1.0** - ***11*** - *Treinamento distribuido*
"""

# Commented out IPython magic to ensure Python compatibility.
try:
#   %tensorflow_version 2.x
except:
  pass
import tensorflow as tf
import numpy as np
import time

# carregando base de dados MNIST
(x_training, y_training), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# normalização da base de dados
x_training = x_training / 255.
x_test = x_test / 255.

# mudança nas dimensões
x_training = x_training.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# definindo modelo (não distribuido)
model_normal = tf.keras.models.Sequential()
model_normal.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))
model_normal.add(tf.keras.layers.Dropout(rate=0.2))
model_normal.add(tf.keras.layers.Dense(units=10, activation='softmax'))
# compilando modelo normal
model_normal.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['sparse_categorical_accuracy'])
model_normal.summary()

# definindo modelo (distribuido)
distribute = tf.distribute.MirroredStrategy()
with distribute.scope():
  model_distributed = tf.keras.models.Sequential()
  model_distributed.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))
  model_distributed.add(tf.keras.layers.Dropout(rate=0.2))
  model_distributed.add(tf.keras.layers.Dense(units=10, activation='softmax'))
  # compilando modelo distribuido
  model_distributed.compile(optimizer='adam', 
                            loss='sparse_categorical_crossentropy', 
                            metrics=['sparse_categorical_accuracy'])

model_distributed.summary()

# comparando a velocidade entre o treinamento normal e distribuido
starting_time = time.time()
model_distributed.fit(x_training, y_training, epochs= 20, batch_size= 128)
print(f'\nDistributed training: {time.time() - starting_time}')

starting_time = time.time()
model_normal.fit(x_training, y_training, epochs= 20, batch_size= 128)
print(f'\nDistributed training: {time.time() - starting_time}')