import random
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D
from keras.datasets import cifar10
from keras.utils import to_categorical


def prepare_data(dataset):
    """
    loads a given dataset, splits into X (images) and y (labels),
    selects only the first 10000 images, splits into train and test-set

    Params:
        input: a keras dataset
        output: x_train, y_train, x_test, y_test
    """
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
    train_images = train_images[:10000] / 255
    test_images = test_images[:10000] / 255
    train_labels = to_categorical(train_labels, 10)[:10000]
    test_labels = to_categorical(test_labels, 10)[:10000]

    return (train_images, train_labels), (test_images, test_labels)


def create_model(network):
    """
    generates the actual model

    Params:
      input: dictionary of parameters
      output: model containing a Pooling and a flattening layer + output for no_classes
    """

    parameters = network._params

    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))

    counter = 0
    for layer in parameters["layers_2D
        counter += 1
        model.add(
            Conv2D(
                filters=parameters["layers_2D"].get(layer).get("units"),
                kernel_size=parameters["layers_2D"].get(layer).get("kernel"),
                strides=parameters["layers_2D"].get(layer).get("stride"),
                padding=parameters["layers_2D"].get(layer).get("padding"),
                activation=parameters["layers_2D"].get(layer).get("activation"),
            )
        )

        model.add(Dropout(parameters["layers_2D"].get(layer).get("dropout")))
        if counter % 3 == 0 or counter == len(parameters['layers_2D'].keys()):
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())

    for layer in parameters["layers_dense"]:
        model.add(
            Dense(
                units=parameters["layers_dense"].get(layer).get("units"),
                activation=parameters["layers_dense"].get(layer).get("activation"),
            )
        )
        model.add(Dropout(parameters["layers_dense"].get(layer).get("dropout")))

    model.add(Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION))
    model.compile(
        loss=LOSS, optimizer=parameters.get("optimizer"), metrics=["accuracy"]
    )
    model.fit(
        train_images,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=parameters.get("epochs"),
        verbose=0,
    )

    return model


def assess_networks(networks):
    """
    attempts to build models from each network and
    generates accuracies by evaluating them on the test-set
    """
    for network in networks:
        try:
            model = create_model(network)
            accuracy = model.evaluate(test_images, test_labels)[1]
            network._accuracy = accuracy
            print("Accuracy: {}".format(network._accuracy))

        except:
            network._accuracy = 0
            print("Build failed.")
    return networks


def select_best_networks(networks):
    """
    sorts models by best performance and keeps only top 20%
    """
    networks = sorted(networks, key=lambda network: network._accuracy, reverse=True)
    networks = networks[: int(0.2 * len(networks))]

    return networks
