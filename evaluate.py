# evaluate.py

import tensorflow as tf
from data.fashion_mnist_data import load_data

# Load Fashion MNIST data
(x_train, y_train), (x_test, y_test) = load_data()

# Load the trained model
model = tf.keras.models.load_model('cnn_fashion_mnist_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
