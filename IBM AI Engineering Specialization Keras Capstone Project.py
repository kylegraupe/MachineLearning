"""This script uses Tensorflow 2.7 with GPU support to train a VGG16 image classifier and compare its results
to a pre-trained ResNet50 image classifier. Follow the instructions in the comments for proper use."""

# Part 1.1: Import Libraries
import tensorflow.keras.models
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import preprocess_input
import numpy as np
import wget
import zipfile
import time
from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


def get_data():
    """
    Call this function to get the concrete_data_week4.zip file and unzip it to the proper directory. Make
    sure to chang the path of the "output" variable to the path you would like to use. Only call this once in order to
    speed up the process.
    """

    url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week4.zip'
    output = r'C:\Users\Kyle\PycharmProjects\pythonProject\Keras Capstone'
    filename = wget.download(url, out=output)

    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(output)


def create_vgg16():
    """
    PART 1: Use this function to create a new VGG16 classifier. Only call it once in the main method for speed. The
    model will be saved in the directory for future use.
    """

    num_classes = 2
    image_resize = 224

    # Part 1.2: Use batch size of 100 images for both training and validation
    train_batch = 100
    valid_batch = 100

    # Part 1.3: Construct ImageDataGenerator for the training set and validation set. Set target size to image_resize.
    data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    train_generator = data_generator.flow_from_directory(
        r'C:\Users\Kyle\PycharmProjects\pythonProject\Keras Capstone\concrete_data_week4\train',
        target_size=(image_resize, image_resize),
        batch_size=train_batch,
        class_mode='categorical')

    validation_generator = data_generator.flow_from_directory(
        r'C:\Users\Kyle\PycharmProjects\pythonProject\Keras Capstone\concrete_data_week4\valid',
        target_size=(image_resize, image_resize),
        batch_size=valid_batch,
        class_mode='categorical')

    # Part 1.4: Create a sequential model. Add VGG16 to it and a Dense layer for output.
    model = Sequential()

    model.add(VGG16(
        include_top=False,
        pooling='avg',
        weights='imagenet',
        ))

    model.add(Dense(num_classes, activation='softmax'))

    print(model.layers[0].layers)

    # We do not want the VGG16 to be retrained, so we write the following line to make sure it only trains the
    # output Dense layer.
    model.layers[0].trainable = False

    # View the model details
    model.summary()

    # Part 1.5: Compile model using 'adam' optimizer and 'categorical_crossentropy' loss function.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    steps_per_epoch_training = len(train_generator)
    steps_per_epoch_validation = len(validation_generator)
    num_epochs = 10

    # Part 1.6: Fit the model on the data.
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch_training,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=steps_per_epoch_validation,
        verbose=1, callbacks=[tensorboard_callback]
    )

    model.save('classifier_VGG16_model.h5')


def eval_model():
    """
    PART 2: Call this model to evaluate the saved ResNet50 and VGG16 models. This function will print out the evaluation results.
    It also will return the models and the test generator for use in the "predictions" function.
    """

    image_resize = 224
    # Part 2.1: Load saved model that was built using ResNet50 model.
    new_resnet = tensorflow.keras.models.load_model('classifier_resnet_model.h5')
    new_vgg16 = tensorflow.keras.models.load_model('classifier_VGG16_model.h5')

    # Part 2.2: Construct an ImageDataGenerator for the test set. For this ImageDataGenerator instance, you only need to
    # pass the directory of the test images, target size, and the shuffle parameter and set it to False.
    data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )
    test_generator = data_generator.flow_from_directory(
        r'D:\Pycharm Projects\Keras\Keras Capstone\concrete_data_week4\test',
        target_size=(image_resize, image_resize),
        shuffle=False)

    test_generator.reset()

    # Part 2.3: Use the evaluate_generator method to evaluate your models on the test data, by passing the above
    # ImageDataGenerator as an argument.
    print("=======================================")
    print("The evaluations for the entire dataset using both ResNet50 and VGG16: \n")
    # Part 2.5: Print the performance of the classifier using the ResNet50 pre-trained model.
    print("ResNet50: \n")
    print(new_resnet.evaluate_generator(test_generator))
    # Part 2.4: Print the performance of the classifier using the VGG16 pre-trained model.
    print("\nVGG16: \n")
    print(new_vgg16.evaluate_generator(test_generator))
    return new_resnet, new_vgg16, test_generator


def predictions(resnet_, vgg16_, test_gen_):
    """
    PART 3: This function returns the predictions of the save ResNet50 and VGG16 models. It outputs the predictions of the first
    five images in the test set.
    :param resnet_:
    :param vgg16_:
    :param test_gen_:
    :return:
    """

    # Part 3.1: Use the predict_generator method to predict the class of the images in the test data, by passing the
    # test data ImageDataGenerator instance defined in the previous part as an argument.
    pred_res = resnet_.predict_generator(test_gen_)
    pred_vgg = vgg16_.predict_generator(test_gen_)

    res_indices = np.argmax(pred_res, axis=1)
    vgg_indices = np.argmax(pred_vgg, axis=1)

    labels = test_gen_.class_indices
    labels = dict((x, y) for y, x in labels.items())
    res_predicts = [labels[y] for y in res_indices]
    vgg_predicts = [labels[y] for y in vgg_indices]
    print("=======================================")
    print("The predictions for the entire dataset: \n")
    print("ResNet50: \n")
    print(res_predicts)
    print("\nVGG16: \n")
    print(vgg_predicts)

    # Part 3.2: Report the class predictions of the first five images in the test set.
    print("\n=======================================")
    print("The predictions for the first five images: \n")
    print("ResNet50: \n")
    print(res_predicts[0:5])
    print("\nVGG16: \n")
    print(vgg_predicts[0:5])


if __name__ == '__main__':
    # Uncomment the line below to retrieve the dataset. Otherwise, leave it commented out. Only do this once.
    # get_data()
    # Uncomment the line below to train the VGG16 model. Otherwise, leave it commented out.
    create_vgg16()
    resnet, vgg16, test_gen = eval_model()
    predictions(resnet, vgg16, test_gen)

    # In command prompt, run "tensorboad --logdir logs\fit" and copy the address to view tensorboard
    # output in a browser. make sure to run in the current directory. "D:\Pycharm Projects\Keras\Keras Capstone"
