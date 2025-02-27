import tensorflow as tf

height = tf.constant(170, dtype='float32')
shoes_size = tf.constant(260, dtype='float32')

# shoes_size_predict = (height * weight) + bias
weight = tf.Variable(0.1)
bias = tf.Variable(0.2)

def loss_function():
    shoes_size_predict = (height * weight) + bias
    error = tf.square(shoes_size - shoes_size_predict)
    return  error

opt = tf.keras.optimizers.Adam(learning_rate=0.2)
opt.minimize(loss_function, var_list=[weight, bias])

shoes_size_predict = (height * weight) + bias
print(shoes_size_predict)
pass