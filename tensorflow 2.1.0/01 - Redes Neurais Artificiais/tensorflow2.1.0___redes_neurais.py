import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist # https://www.kaggle.com/zalando-research/fashionmnist

# Separando base de dados
(x_training, y_training), (x_test, y_test) = fashion_mnist.load_data()
print(' - Shapes:')
print(f'x_training.shape -> {x_training.shape}')
print(f'x_test.shape -> {x_test.shape}')
print(f'y_training.shape -> {y_training.shape}')
print(f'y_test.shape -> {y_test.shape}', end='\n'*2)
print(f' - Dado:\nx_training[0] ->\n{x_training[0]}', end='\n'*2)
print(f' - Dado:\nnp.unique(y_training) -> {np.unique(y_training)}', end='\n'*3)

# Normalizando base de dados (Dividimos cada elemento(pixel) da base de dados por 255(maior valor de um pixel de uma imagem) para obter valores entre 0 e 1
print(' - Normalizando base de dados\n...')
x_training = x_training / 255.0
x_test = x_test / 255.0
print(f' - Dado normalizado:\nx_training[0] ->\n{x_training[0]}', end='\n'*3)

# Remodelando base de dados (mudando de matriz para vetor)
print(' - Remodelando base de dados')
x_training = x_training.reshape(-1, 28 * 28)
x_test     = x_test.reshape(-1, 28 * 28)
print(f'x_training.shape -> {x_training.shape}')
print(f'x_test.shape -> {x_test.shape}', end='\n'*3)

# Construindo a rede neural
neuronios_entrada = 784
neuronios_oculta = 128
neuronios_saida = 10
taxa_dropout = 0.2

# Criando o modelo
model = tf.keras.models.Sequential()
# Camada de entrada e camada oculta 1
model.add(tf.keras.layers.Dense(units= neuronios_oculta, activation= 'relu', input_shape= (neuronios_entrada, )))
# Adicionando Dropout 1
model.add(tf.keras.layers.Dropout(taxa_dropout))
# Adicionando mais camadas ocultas
model.add(tf.keras.layers.Dense(units= neuronios_oculta, activation= 'relu')) # camada oculta 2
model.add(tf.keras.layers.Dropout(taxa_dropout)) # Dropout 2
model.add(tf.keras.layers.Dense(units= neuronios_oculta, activation= 'relu')) # camada oculta 3
model.add(tf.keras.layers.Dropout(taxa_dropout)) # Dropout 3
model.add(tf.keras.layers.Dense(units= neuronios_oculta, activation= 'relu')) # camada oculta 4
model.add(tf.keras.layers.Dropout(taxa_dropout)) # Dropout 4
# Camada de saida
model.add(tf.keras.layers.Dense(units= neuronios_saida, activation= 'softmax'))
# Compilando o modelo ('sparse porque a saida não é no formato OneHotEncode')
model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics= ['sparse_categorical_accuracy'])
print(' - Resumo do modelo')
model.summary()

# Treinamento do modelo
print('\n'*2 + ' - Treinamento do modelo')
model.fit(x_training, y_training, epochs= 20)

# Avaliando o modelo
print('\n'*2 + ' - Avaliando o modelo')
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Taxa de acerto -> {test_accuracy}')

# Salvando o modelo
print('\n'*2 + ' - Salvando o modelo ...')
model_json = model.to_json()
with open('fashion_model.json', 'w') as json_file:
    json_file.write(model_json)
    print('Modelo salvo!')

# Salvando os pesos
print('\n'*2 + ' - Salvando os pesos do modelo ...')
model.save_weights('fashion_model.h5')
print('Pesos do modelos salvos!')