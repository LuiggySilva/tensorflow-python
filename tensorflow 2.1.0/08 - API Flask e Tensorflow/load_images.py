from tensorflow.keras.datasets import fashion_mnist
from scipy.misc import imsave

(x_training, y_training), (x_test, y_test) = fashion_mnist.load_data()

for i in range(5):
    imsave(name= f'uploads/{i}.png', arr= x_test[i])