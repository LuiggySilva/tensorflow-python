# -*- coding: utf-8 -*-
"""tensorflow2.1.0___tensorflow_serving.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G6c_hZ5gvADP2l4V8ZZdpTIODNtB3ce7

> # **TensorFlow  2.1.0** - ***09*** - *Tensorflow Serving*
"""

!pip install tensorflow-gpu==1.13.1

!echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

!apt-get update && apt-get install tensorflow-model-server

!pip install requests

# Commented out IPython magic to ensure Python compatibility.
import os
import json
import random
import requests
import subprocess
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

# %matplotlib inline
tf.__version__

# pré-processamento da base de dados
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

X_train = X_train / 255.0
X_test = X_test / 255.0

# definição do modelo
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.summary()

# compilando o modelo
model.compile(optimizer='Adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['sparse_categorical_accuracy'])

# treinando o modelo
model.fit(X_train, 
          y_train, 
          batch_size=128, 
          epochs=10)

# avaliando o modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy is {}".format(test_accuracy))

# salvando o modelo para produção
model_dir = 'model/'
version = 1

export_path = os.path.join(model_dir, str(version))
if(os.path.isdir(export_path)):
  !rm -r {export_path}

tf.saved_model.simple_save(tf.keras.backend.get_session(), export_dir = export_path,
                           inputs = {"input_image": model.input},
                           outputs = {t.name: t for t in model.outputs})

# configurando o ambiente de produção
os.environ["model_dir"] = os.path.abspath(model_dir)

# Commented out IPython magic to ensure Python compatibility.
# %%bash --bg
# nohup tensorflow_model_server --rest_api_port=8501 --model_name=cifar10 --model_base_path="${model_dir}" >server.log 2>&1

!tail server.log

# realizando requisição para o servidor
random_image = np.random.randint(0, len(X_test))

data = json.dumps({'signature_name': 'serving_default', 'instances': [X_test[random_image].tolist()]})
headers = {'content-type': 'application/json'}
json_response = requests.post(url= 'http://localhost:8501/v1/models/cifar10:predict', data= data, headers= headers)

print(json_response)

predictions = json.loads(json_response.text)['predictions']
print(predictions)

plt.imshow(X_test[random_image])
plt.title(f'Class predicted: {class_names[np.argmax(predictions)]}', fontdict={'fontsize':20})
plt.axis('off')

# requisição para versão especifica de um modelo
random_image = np.random.randint(0, len(X_test))
data = json.dumps({'signature_name': 'serving_default', 'instances': [X_test[random_image].tolist()]})
especific_json_response = requests.post(url= 'http://localhost:8501/v1/models/cifar10/versions/1:predict', data= data, headers= headers)
print(especific_json_response)

especific_predictions = json.loads(especific_json_response.text)['predictions']
print(especific_predictions)

plt.imshow(X_test[random_image])
plt.title(f'Class predicted: {class_names[np.argmax(especific_predictions)]}', fontdict={'fontsize':20})
plt.axis('off')