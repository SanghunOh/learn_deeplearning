#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, models, layers, optimizers, metrics
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications import MobileNetV3Large

np.random.seed(3)
tf.compat.v1.set_random_seed(3)

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
       class_mode='categorical'
       )

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
       './test',
       target_size=(50, 50),
       batch_size=5,
       class_mode='categorical'
       )

# transfer_model = VGG16(weights='imagenet', include_top=False, input_shape=(50, 50, 3))
transfer_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(50, 50, 3))
transfer_model.trainable = False
transfer_model.summary()

finetune_model = models.Sequential()
finetune_model.add(transfer_model)
finetune_model.add(Flatten())
finetune_model.add(Dense(64, activation='relu'))
finetune_model.add(Dense(4, activation='softmax'))
finetune_model.summary()

finetune_model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0002), metrics=['accuracy'])

finetune_model.save('./models/my_model.h5')

history = finetune_model.fit(
       train_generator,
       steps_per_epoch=100,
       epochs=20,
       validation_data=test_generator,
       validation_steps=4)

# acc= history.history['accuracy']
# val_acc= history.history['val_accuracy']
# y_vloss = history.history['val_loss']
# y_loss = history.history['loss']


# 그래프로 표현
# x_len = np.arange(len(y_loss))
# plt.plot(x_len, acc, marker='.', c="red", label='Trainset_acc')
# plt.plot(x_len, val_acc, marker='.', c="lightcoral", label='Testset_acc')
# plt.plot(x_len, y_vloss, marker='.', c="cornflowerblue", label='Testset_loss')
# plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# # 그래프에 그리드를 주고 레이블을 표시
# plt.legend(loc='upper right')
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('loss/acc')
# plt.show()
