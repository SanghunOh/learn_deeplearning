"""## load MNIST data"""

import keras
from keras.datasets import mnist
from keras.models import Sequential

from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train.shape

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train sample')
print(x_test.shape[0], 'x_test sample')

num_classes = 10

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('x_train', x_train[0])
print('y_train', y_train[0])

"""## Create Model"""

model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(784,))) # input layer
model.add(Dense(512, activation='relu')) # hidden layer1
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.00001), metrics=['accuracy'])

model.summary()
