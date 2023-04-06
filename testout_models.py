
from keras.models import load_model
import numpy as np
model = load_model('h5 files/cnn_model.h5')
test_data = np.load('test_data_112.npy')
test_images = test_data[:, :-12].reshape(-1, 112, 112, 3)
test_images = test_images / 255.0
#features = model.predict(test_images)
#test_data = np.expand_dims(test_data[:, -12:], axis=(1, 2))
#print(test_data.shape)
test_loss, test_acc = model.evaluate(test_images, test_data[:, -12:])
print('Test accuracy:', test_acc)

