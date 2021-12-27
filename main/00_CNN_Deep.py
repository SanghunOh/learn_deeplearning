#!/usr/bin/env python
#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)

# 데이터 불러오기
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                  horizontal_flip=True,
                                  width_shift_range=0.5,
                                  height_shift_range=0.5,
                                  vertical_flip=True,
                                  zoom_range=20,rotation_range=0.5,
                                  fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
       './train',
       target_size=(50, 50),
       batch_size=5,
       class_mode='sparse'
       )

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
       './test',
       target_size=(50, 50),
       batch_size=5,
       class_mode='sparse'
       )

# 컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(50, 50, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,  activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 최적화 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath="./models/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

LOGS_DIR = './logs/'
if not os.path.exists(LOGS_DIR):
   os.makedirs(LOGS_DIR)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR, histogram_freq=1, write_graph=True,write_images=True)

# 모델의 실행
# history = model.fit(train_generator, steps_per_epoch=100, validation_data=test_generator, epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer,tensorboard_callback])
# history = model.fit(train_generator, steps_per_epoch=100, validation_data=test_generator, epochs=20, batch_size=200, callbacks=[tensorboard_callback])
history = model.fit(
       train_generator,
       steps_per_epoch=100,
       epochs=20,   # epochs=20,
       validation_data=test_generator,
       validation_steps=4,
       callbacks=[tensorboard_callback,],)

model.save('./models/my_model.h5')
model.save('./models/my_model')
# 테스트 정확도 출력
# print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
print("\n Test Accuracy: %.4f" % (model.evaluate(test_generator, steps=5)[1]))

# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

