import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras_preprocessing.image import ImageDataGenerator
from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples
import keras
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D
from keras.layers import Flatten
from keras.models import Sequential
from matplotlib import pyplot as plt
from kerastuner import BayesianOptimization

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
def model_builder(hp):
    num_classes = 47
    # build a sequential model
    model = Sequential()
    # 1st conv block
    hp_firstConv = hp.Int('filter_1', min_value=32, max_value=128, step=32)
    model.add(Convolution2D(filters=hp_firstConv, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1),
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # 2nd conv block
    hp_secondConv = hp.Int('filter_2', min_value=32, max_value=128, step=32)
    model.add(Convolution2D(filters=hp_secondConv, kernel_size=(3, 3), activation='relu'))
    model.add(Convolution2D(filters=hp_secondConv, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # 3nd conv block
    hp_thirdConv = hp.Int('filter_3', min_value=32, max_value=128, step=32)
    model.add(Convolution2D(filters=hp_thirdConv, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # flatten output of conv
    model.add(Flatten())
    # hidden layer
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(Dense(units=hp_units, activation='relu'))
    model.add(Dense(128, activation="relu"))
    # output layer
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, default=0.25, step=0.05)
    model.add(Dropout(hp_dropout))
    model.add(Dense(num_classes, activation='softmax'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=['accuracy'])
    return model


# endregion

# region Tuner Model training
tuner = BayesianOptimization(model_builder, objective='val_accuracy', max_trials=20)
tuner.search(training_images, training_labels, validation_data=(test_images, test_labels), epochs=10)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal numbers are:\n
First Convolution layer is {best_hps.get('filter_1')}\n
Second Convolution layer is {best_hps.get('filter_2')}\n
Third Convolution layer is {best_hps.get('filter_3')}\n
Dropout Layer is {best_hps.get('dropout')}
Layer Dense is {best_hps.get('units')}\n
Learning rate is {best_hps.get('learning_rate')}.
""")
# Build the model with the optimal hyperparameters and train it on the data for 10 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(training_images, training_labels, epochs=10, validation_data=(test_images, test_labels),
                    shuffle=True, verbose=2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
print("Starting training the final model...")
history = hypermodel.fit(training_images, training_labels, epochs=best_epoch,
                         validation_data=(test_images, test_labels))
print("Training model successfully...")
eval_result = hypermodel.evaluate(test_images, test_labels)
print("[test loss, test accuracy]:", eval_result)

# save the model
model.save('hypermodel.h5')
print('The model has sucessfully been saved...')
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
