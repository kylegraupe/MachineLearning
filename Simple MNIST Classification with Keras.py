"""This script is a simple classification example using Keras on the MNIST dataset."""

# Import required libraries.
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.datasets import mnist # import the data
from keras.models import load_model


# read the data
def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return (X_train, y_train), (X_test, y_test)


# define classification model
def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model




if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_data()

    print(X_train.shape)

    plt.imshow(X_train[0])

    # flatten images into one-dimensional vector
    num_pixels = X_train.shape[1] * X_train.shape[2]  # find size of one-dimensional vector

    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')  # flatten training images
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')  # flatten test images

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encode outputs
    y_train = tensorflow.keras.utils.to_categorical(y_train)
    y_test = tensorflow.keras.utils.to_categorical(y_test)

    num_classes = y_test.shape[1]
    print(num_classes)
    # build the model
    model = classification_model()

    # fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

    # evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=0)

    print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))

    model.save('classification_model.h5')

    pretrained_model = load_model('classification_model.h5')