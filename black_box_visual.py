import numpy as np
from keras.models import load_model
from skimage import io
from skimage.util import img_as_float
from skimage.transform import resize
import tensorflow as tf
import matplotlib.pyplot as plt

# Load your trained model
model = load_model('h5 files/fcn_model.h5')
#model = load_model('h5 files/xception_model.h5')

# Load and preprocess the input image
#image_path = 'C:/Users/18589/Desktop/ECE176 project/battery/battery34.jpg'
image_path = 'C:/Users/18589/Desktop/ECE176 project/white-glass/white-glass515.jpg'
image = io.imread(image_path)
image = img_as_float(image)
image = resize(image, (112, 112))
x = np.expand_dims(image, axis=0)

# Convert the NumPy array to a TensorFlow tensor
x_tensor = tf.convert_to_tensor(x)

# Get the model's output for the input image
output = model(x_tensor)

# Compute the gradient of the output with respect to the input
with tf.GradientTape() as tape:
    tape.watch(x_tensor)
    output = model(x_tensor)
grads = tape.gradient(output, x_tensor)

# Calculate the saliency map
saliency_map = np.max(np.abs(grads), axis=-1)[0]

# Normalize the saliency map
saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))

# Plot the original image and the saliency map
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image)
ax1.set_title('Original Image')
ax2.imshow(saliency_map, cmap='hot')
ax2.set_title('Saliency Map')
plt.show()