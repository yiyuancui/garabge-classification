from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure
from keras.models import load_model
import cv2
# Load the trained model from an HDF5 file
model = load_model('xception_model.h5')
#layer_name = 'block13_pool'
layer_name = 'conv2d_2'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
#train_data = np.load('train_data.npy')
#train_images = train_data[:, :-12].reshape(-1, 112, 112, 3)
#intermediate_output = intermediate_layer_model.predict(train_images[:32])

# ## mask heatmap

# n = 31
# mask = np.zeros((112, 112))
# for i in range(1024):
#     feature_map = intermediate_output[n, :, :, i]
#     feature_map = (feature_map * 255).astype(np.uint8)
#     feature_map = cv2.resize(feature_map, (112, 112))
#     mask += feature_map
#
# train_image = train_images[n]
# train_image = (train_image * 255).astype(np.uint8)
# train_image = cv2.resize(train_image, (512, 512))
#
# mask = mask / np.max(mask)
# heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
# heatmap = cv2.resize(heatmap, (512, 512))
#
# result = cv2.addWeighted(train_image, 0.6, heatmap, 0.4, 0)
#
# cv2.imshow("Training Image", train_image)
# cv2.imshow("Activation Map", heatmap)
# cv2.imshow("Result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## countour

#intermediate_output = intermediate_layer_model.predict(train_images)


image_path = "C:/Users/18589/Desktop/ECE176 project/white-glass/white-glass515.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (112, 112))
image = np.expand_dims(image, axis=0)
image = image.astype('float32') / 255

intermediate_output = intermediate_layer_model.predict(image)

# Get the mean of the intermediate output across all training images
mean_output = np.mean(intermediate_output, axis=0)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax1.imshow(image[0])
# ax2.contourf(mean_output[:,:,0], cmap='Reds', alpha=0.5)
# ax2.contourf(mean_output[:,:,1], cmap='Greens', alpha=0.5)
# ax2.contourf(mean_output[:,:,2], cmap='Blues', alpha=0.5)
# plt.show()


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

