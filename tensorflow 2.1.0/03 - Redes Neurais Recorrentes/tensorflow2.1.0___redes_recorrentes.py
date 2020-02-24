import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import random, numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb # https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Pre-processamento dos dados
number_of_words = 20000 # Para todos textos da base de dados
max_len = 100 # Para cada texto da base

(x_training, y_training), (x_test, y_test) = imdb.load_data(num_words= number_of_words)

print(' - Shapes:')
print(f'x_training.shape -> {x_training.shape}')
print(f'x_test.shape -> {x_test.shape}')
print(f'y_training.shape -> {y_training.shape}')
print(f'y_test.shape -> {y_test.shape}', end='\n'*2)
print(f' - Dado:\nx_training[0] ->\n{x_training[0]}', end='\n'*2)
print(f' - Dado:\nnp.unique(y_training) -> {np.unique(y_training)}', end='\n'*3)

# Visualizando tamanho dos dados
print(' - Lens:')
print(f'len(x_training[0]) -> {len(x_training[0])}')
print(f'len(x_training[1]) -> {len(x_training[1])}', end='\n'*2)

# Normalizando dados
print(' - Normalizando dados:')
x_training = tf.keras.preprocessing.sequence.pad_sequences(x_training, maxlen= max_len)
x_test     = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen= max_len)
print(f'len(x_training[0]) -> {len(x_training[0])}')
print(f'len(x_training[1]) -> {len(x_training[1])}')
print(f'x_training.shape -> {x_training.shape}')
print(f'x_test.shape -> {x_test.shape}', end='\n'*2)

# Construindo a rede neural
model = tf.keras.Sequential()
# Adicionando a camada de Embedding
model.add(tf.keras.layers.Embedding(input_dim= number_of_words, output_dim= 128, input_shape= (x_training.shape[1], )))
# Adicionando a camadas de LSTM
model.add(tf.keras.layers.LSTM(units= 128, activation= 'tanh'))
# Adicionando camada de saida
model.add(tf.keras.layers.Dense(units= 1, activation= 'sigmoid'))

# Compilando o modelo
model.compile(optimizer= 'rmsprop',loss= 'binary_crossentropy',metrics= ['accuracy'])
print(' - Resumo do modelo')
model.summary()

# Treinamento do modelo
print('\n'*2 + ' - Treinamento do modelo')
model.fit(x_training, y_training, epochs= 5, batch_size= 128)

# Avaliando o modelo
print('\n'*2 + ' - Avaliando o modelo')
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Taxa de acerto -> {test_accuracy}') 
print(f'Taxa de erro -> {test_loss}')

# Salvando o modelo
print('\n'*2 + ' - Salvando o modelo ...')
model_json = model.to_json()
with open('imdb_model.json', 'w') as json_file:
    json_file.write(model_json)
    print('Modelo salvo!')

# Salvando os pesos
print('\n'*2 + ' - Salvando os pesos do modelo ...')
model.save_weights('imdb_model.h5')
print('Pesos do modelos salvos!')