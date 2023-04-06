import numpy as np
from keras.models import load_model
from skimage import io, color, filters, measure
import matplotlib.pyplot as plt
from keras.models import Model
from skimage.util import img_as_float
from skimage.transform import resize

'''
This file will show us certain output from each layers with the model we trained
i.e. fcn_model which was trained and contruct by myself. And transfer learning using
Xception and deconv layers and train the weight on deconv layers. This will also 
show the things learned by FCN. And in this visualization file. We will see black box
in neural networks. Where early convolutions with low level information gives us the shape of the 
object and the mid and high level information are very hard to be interpreted by human beings.
'''


# Load the trained model from an HDF5 file
model = load_model('h5 files/fcn_model.h5')
#model = load_model('h5 files/xception_model.h5')
layer_name = 'conv2d_3'
#layer_name = 'block13_sepconv2'
#layer_name = 'block14_sepconv2'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

image_path = "C:/Users/18589/Desktop/ECE176 project/white-glass/white-glass515.jpg"
image = io.imread(image_path)
image = img_as_float(image)
image = resize(image, (112, 112))
image = np.expand_dims(image, axis=0)
intermediate_output = intermediate_layer_model.predict(image)

mean_output = np.mean(intermediate_output, axis=0)
if mean_output.shape[-1] > 3:
    mean_output = mean_output[..., :3]

gray = color.rgb2gray(mean_output)
thresh = filters.threshold_otsu(gray)
binary = gray > thresh

contours = measure.find_contours(binary, 0.5)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(image[0])
ax2.contourf(mean_output[:,:,0], cmap='Reds', alpha=0.5)
ax2.contourf(mean_output[:,:,1], cmap='Greens', alpha=0.5)
ax2.contourf(mean_output[:,:,2], cmap='Blues', alpha=0.5)


for cnt in contours:
    ax2.plot(cnt[:, 1], cnt[:, 0], linewidth=2, color='black')
heatmap = np.zeros(binary.shape + (3,))
heatmap[binary] = (1, 0, 0)
ax3.imshow(heatmap)

plt.show()