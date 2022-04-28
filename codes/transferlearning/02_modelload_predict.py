
import tensorflow as tf

new_model = tf.keras.models.load_model('./models/my_model.h5')

# 모델 구조를 출력합니다
new_model.summary()

list = ['accordion', 'airplanes', 'anchor', 'ant']
for dir in list:
       path = './test/'+dir+'/image_0008.jpg'
       image = tf.keras.preprocessing.image.load_img(path, target_size=(150,150))			# with format information
       input_arr = tf.keras.preprocessing.image.img_to_array(image)	# return numpy
       x_pred = input_arr.reshape(-1, 150, 150, 3)
       pred = new_model.predict(x_pred)
       print(dir, ' : ', pred.argmax())

