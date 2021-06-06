import os
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras_preprocessing.image import ImageDataGenerator
from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D
from keras.layers import Flatten
from keras.models import Sequential
from matplotlib import pyplot as plt

# region Data Preparation
print("\nPOSSIBLE EMNIST datasets: " + str(list_datasets()) + "\n We will use 'balanced'...")
print("\nTRAININGDATA\n_______________________________________\n")
print("Loading training data...")
training_images, training_labels = extract_training_samples('balanced')
print("Images shape: " + str(training_images.shape))
print("Labels shape: " + str(training_labels.shape))
print("\nTESTDATA\n_______________________________________\n")
print("Loading test data...")
test_images, test_labels = extract_test_samples('balanced')
print("Images shape: " + str(test_images.shape))
print("Labels shape: " + str(test_labels.shape))

# flatten 28*28 images to a 784 vector for each image
num_pixels = training_images.shape[1] * training_images.shape[2]

# reshape format [samples] [width] [height] [channels]
training_images = training_images.reshape(training_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

# print a few images
plt.imshow(training_images[0], cmap='gray', interpolation='none')
plt.show()
plt.imshow(training_images[20000], cmap='gray', interpolation='none')
plt.show()
plt.imshow(training_images[30000], cmap='gray', interpolation='none')
plt.show()
plt.imshow(training_images[45000], cmap='gray', interpolation='none')
plt.show()

# print after reshape
print("\nTRAININGDATA AFTER RESHAPE\n_______________________________________\n")
print("Images shape: " + str(training_images.shape))
print("\nTESTDATA AFTER RESHAPE\n_______________________________________\n")
print("Images shape: " + str(test_images.shape))

# DATA PREPARATION
print("Rotating and shifting images....")
datagen = ImageDataGenerator(rotation_range=45, width_shift_range=0.2, height_shift_range=0.2, shear_range=30,
                             zoom_range=[0.5, 1.5])
print("Rotating and shifting images sucessfully....")
# fit parameters from data
datagen.fit(training_images)

# converts a class vector (integers) to binary class matrix, one hot encoding
training_labels = keras.utils.to_categorical(training_labels)
test_labels = keras.utils.to_categorical(test_labels)

# normalize inputs
training_images = training_images / 255
test_images = test_images / 255


# endregion

# region Define a CNN model
# 26 + 26 + 10 = (62 für 'byclass'), (47 für 'bymerge', (digits, mnist = 10) , (Letters = 26), sonst 47
def create_model():
    num_classes = 47
    # build a sequential model
    model = Sequential()
    # 1st conv block
    model.add(Convolution2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1), padding='same'))
    model.add(Convolution2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.1))
    # 2nd conv block
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.2))
    # 3nd conv block
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.3))
    # flatten output of conv
    model.add(Flatten())
    # hidden layer
    model.add(Dense(64, activation="relu"))
    model.add(Dense(128, activation="relu"))
    # output layer
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# endregion

# region Model creation
print("\nCreating model...")
model = create_model()
print("Model has been successfully created...")
print("Training model...")
# fit the model
history = model.fit(training_images, training_labels, validation_data=(test_images, test_labels), epochs=20,
                    batch_size=256, shuffle=True, verbose=2)
print("Model has been successfully trained...")

# save the model
model.save('model_ReadyToUse.h5')
print('The model has sucessfully been saved...')

# evaluate the model
scores = model.evaluate(test_images, test_labels, verbose=0)
model.summary()
# endregion

# region Plots
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# endregion
