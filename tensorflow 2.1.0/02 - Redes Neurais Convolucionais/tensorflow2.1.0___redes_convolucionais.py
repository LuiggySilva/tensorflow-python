import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import random, numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10 # https://www.cs.toronto.edu/~kriz/cifar.html

# Separando base de dados
(x_training, y_training), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(' - Shapes:')
print(f'x_training.shape -> {x_training.shape}')
print(f'x_test.shape -> {x_test.shape}')
print(f'y_training.shape -> {y_training.shape}')
print(f'y_test.shape -> {y_test.shape}', end='\n'*2)
print(f' - Dado:\nx_training[0] ->\n{x_training[0]}', end='\n'*2)
print(f' - Dado:\nnp.unique(y_training) -> {np.unique(y_training)}', end='\n'*2)
print(f' - Nome das classes -> {class_names}', end='\n'*2)

# Normalizando base de dados (Dividimos cada elemento(pixel) da base de dados por 255(maior valor de um pixel de uma imagem) para obter valores entre 0 e 1
print(' - Normalizando base de dados\n...')
x_training = x_training / 255.0
x_test = x_test / 255.0
print(f' - Dado normalizado:\nx_training[0] ->\n{x_training[0]}', end='\n'*3)

# Exemplo da base de dados
#fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(15, 7))
#axs = list(axs)
##    classes 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
#img_indexs = [49,45,18,39,28,27,22,12,8,2] 
#
#index = 0
#for line_axs in axs:
#  for i in range(len(line_axs)):
#    ax = line_axs[i]
#    ax.imshow(x_training[img_indexs[index]].reshape((32,32, 3)))
#    ax.axis('off')
#    index += 1

# Construindo a rede neural
neuronios_entrada = (32, 32, 3)
neuronios_oculta = 128
neuronios_saida = 10

# Criando o modelo
model = tf.keras.models.Sequential()
# Camadas de Convolução e MaxPooling 
model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= 3, padding= 'same', activation= 'relu', input_shape= neuronios_entrada)) # Convoluçao 1
model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= 3, padding= 'same', activation= 'relu')) # Convoluçao 2
model.add(tf.keras.layers.MaxPooling2D(pool_size= 2, strides= 2, padding='valid')) # MaxPolling 1
model.add(tf.keras.layers.Conv2D(filters= 64, kernel_size= 3, padding= 'same', activation= 'relu')) # Convoluçao 3
model.add(tf.keras.layers.Conv2D(filters= 64, kernel_size= 3, padding= 'same', activation= 'relu')) # Convoluçao 4
model.add(tf.keras.layers.MaxPooling2D(pool_size= 2, strides= 2, padding='valid')) # MaxPolling 2
# Camada de Flatting
model.add(tf.keras.layers.Flatten())
# Camada de entrada
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(units= neuronios_oculta, activation= 'relu'))
model.add(tf.keras.layers.Dense(units= neuronios_oculta, activation= 'relu'))
# Camada de saida
model.add(tf.keras.layers.Dense(units= neuronios_saida, activation= 'softmax'))

print(' - Resumo do modelo')
model.summary()

# Compilando modelo
model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics= ['sparse_categorical_accuracy'])

# Treinamento do modelo
print('\n'*2 + ' - Treinamento do modelo')
model.fit(x_training, y_training, epochs= 5)

# Avaliando o modelo
print('\n'*2 + ' - Avaliando o modelo')
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Taxa de acerto -> {test_accuracy}')

# Salvando o modelo
print('\n'*2 + ' - Salvando o modelo ...')
model_json = model.to_json()
with open('cifar10_model.json', 'w') as json_file:
    json_file.write(model_json)
    print('Modelo salvo!')

# Salvando os pesos
print('\n'*2 + ' - Salvando os pesos do modelo ...')
model.save_weights('cifar10_model.h5')
print('Pesos do modelos salvos!')