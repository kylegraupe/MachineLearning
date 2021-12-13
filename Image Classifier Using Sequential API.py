"""This script shows a simple image classifier on the MNIST Fashion dataset using a Keras Sequential API."""

from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


def create_val_set(X_train_full_, y_train_full_):
    """This function creates the training and validation datasets."""
    X_valid_, X_train_ = X_train_full_[:5000] / 255.0, X_train_full_[5000:] / 255.0
    y_valid_, y_train_ = y_train_full_[:5000], y_train_full_[5000:]
    return X_valid_, X_train_, y_valid_, y_train_


def class_names():
    """This function gives descriptive names to the classes in the MNIST Fashion dataset."""
    names_ = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    return names_


def create_sequential_model():
    """This function creates the Sequential model with two hidden layers, and an output layer with 10 neurons, one per
    class. The input shape is determined by the dimensions of the input images in pixels. ReLU activation functions
    are used on the hidden layers and a softmax activation function is used on the output layer. The model is compiled
    using spares categorical cross entropy loss, stochastic gradient descent optimizer, and accuracy metrics."""
    model_ = keras.models.Sequential()
    model_.add(keras.layers.Flatten(input_shape=[28, 28]))
    model_.add(keras.layers.Dense(300, activation='relu'))
    model_.add(keras.layers.Dense(100, activation='relu'))
    model_.add(keras.layers.Dense(10, activation='softmax'))
    model_.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    return model_


def train_model(compiled_model_, X_train_, y_train_, X_valid_, y_valid_):
    """This function trains the model. 30 training epochs are used."""
    trained_model_ = compiled_model_.fit(X_train_, y_train_, epochs=30, validation_data=(X_valid_, y_valid_))
    return trained_model_


def plots(trained_model_):
    """This function plots the performance of the model. It shows the convergence towards a final loss and accuracy."""
    pd.DataFrame(trained_model_.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1) # Set vertical range from 0 to 1.
    plt.show()


if __name__ == '__main__':
    """__main__ function controls the running of the script."""

    # Look at the shape of the training data.
    print(X_train_full.shape)

    # Create validation set instance using the above function. Fashion MNIST does not have a validation set built in.
    X_valid, X_train, y_valid, y_train = create_val_set(X_train_full, y_train_full)

    # Call the class_names() function to store the class names in the below variable.
    names = class_names()

    # Create an instance of the model.
    model = create_sequential_model()

    # Look at model summary.
    print(model.summary())

    # Look at the models layers.
    print(model.layers)

    # To access the weights and biases of a layer (i.e. the first hidden layer), follow this method:
    hidden_1 = model.layers[1]
    weights_1, biases_1 = hidden_1.get_weights()
    # print(hidden1.name, weights_1, biases_1)

    # Train the model.
    trained_model = train_model(model, X_train, y_train, X_valid, y_valid)

    # Plot the performance throughout the model training.
    plots(trained_model)


