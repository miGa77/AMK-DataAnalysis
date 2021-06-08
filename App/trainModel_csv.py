import numpy as np
import pandas as pd
from keras.utils import np_utils
from matplotlib import pyplot as plt

# Constants
IMAGE_SIZE = 28
NUM_CLASSES = 47
print("\nLoading EMNIST Balanced Training dataset...")
train_data = pd.read_csv("./data/emnist-balanced-train.csv", delimiter=",")
print("Done...")
print("Loading EMNIST Balanced Testing dataset...")
test_data = pd.read_csv("./data/emnist-balanced-test.csv", delimiter=",")
print("Done...")
print('Dimension of training data: ', np.shape(train_data))
print('Dimension of testing data: ', np.shape(test_data))
training_images = train_data.iloc[:, 1:]
training_labels = train_data.iloc[:, 0]
test_images = test_data.iloc[:, 1:]
test_labels = test_data.iloc[:, 0]


def rotate(image):
    image = image.reshape([IMAGE_SIZE, IMAGE_SIZE])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image


# Flip and rotate image because they are written like that
training_images = np.asarray(training_images)
training_images = np.apply_along_axis(rotate, 1, training_images)
print("training_images:", training_images.shape)

test_images = np.asarray(test_images)
test_images = np.apply_along_axis(rotate, 1, test_images)
print("test_images:", test_images.shape)

# One hot encoding
training_labels = np_utils.to_categorical(training_labels, NUM_CLASSES)
test_labels = np_utils.to_categorical(test_labels, NUM_CLASSES)
print("training_labels: ", training_labels.shape)
print("test_labels: ", test_labels.shape)
# print a few images
plt.imshow(training_images[0], cmap='gray', interpolation='none')
plt.show()
plt.imshow(training_images[20000], cmap='gray', interpolation='none')
plt.show()
plt.imshow(training_images[30000], cmap='gray', interpolation='none')
plt.show()
plt.imshow(training_images[45000], cmap='gray', interpolation='none')
plt.show()
