
import tensorflow as tf

new_model = tf.keras.models.load_model('./models/my_model.h5')

# 모델 구조를 출력합니다
new_model.summary()

list = ['accordion', 'airplanes', 'anchor', 'ant']
import numpy as np
for dir in list:
       path = './test/'+dir+'/image_0009.jpg'
       image = tf.keras.preprocessing.image.load_img(path, target_size=(50,50))			# with format information
       input_arr = tf.keras.preprocessing.image.img_to_array(image)	# return numpy
       x_pred = input_arr.reshape(-1, 50, 50, 3)
       pred = new_model.predict(x_pred)
       print(dir, ' : ', pred.argmax(), ', ',np.float16(pred))

# result
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                   horizontal_flip=True,
#                                   width_shift_range=0.5,
#                                   height_shift_range=0.5,
#                                   vertical_flip=True,
#                                   zoom_range=20,rotation_range=0.5,
#                                   fill_mode='nearest')
# accordion  :  1 ,  [[2.7504480e-03 8.8278526e-01 8.6723623e-05 1.1437752e-01]]
# airplanes  :  0 ,  [[0.5999716  0.07461271 0.00380612 0.32160965]]
# anchor  :  3 ,  [[0.04411722 0.4071598  0.02920074 0.51952225]]
# ant  :  3 ,  [[0.01742597 0.01553538 0.03536213 0.93167657]]