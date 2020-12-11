#!/usr/bin/env python
# coding: utf-8


import logging
import random
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D
from keras.datasets import cifar10
from keras.utils import to_categorical

from helper_functions import (
    prepare_data,
    create_model,
    assess_networks,
    select_best_networks,
)


# Setup logging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.DEBUG,
    filename="log.txt",
)


### move this to another file
DATASET = cifar10
NUM_CLASSES = 10
BATCH_SIZE = 300
VALIDATION_SPLIT = 0.2
THRESHOLD = 0.8
POPULATION = 10
GENERATIONS = 2
INPUT_SHAPE = (32, 32, 3)
LOSS = "categorical_crossentropy"
OUTPUT_ACTIVATION = "softmax"


class Network:
    """instantiates the networks with randomly chosen parameters"""

    def __init__(self):
        """
        parameters to be optimized:
            2D-layers: units, kernel, stride, padding, dropout, hidden_activation
            dense-layers: units, dropout, hidden_activation
            optimizer, epochs
        """
        self._loss = LOSS
        self._output_activation = OUTPUT_ACTIVATION
        self._accuracy = 0

        self._params = {
            "layers_2D": {},
            "layers_dense": {},
            "optimizer": None,
            "epochs": None,
        }

        placeholder = {
            "activation": None,
            "dropout": None,
            "kernel": None,
            "padding": None,
            "stride": None,
            "units": None,
        }

        for i in range(1, random.randint(2, 8)):
            self._params["layers_2D"].setdefault(str(i), placeholder)

        for i in range(1, random.randint(2, 8)):
            self._params["layers_dense"].setdefault(str(i), placeholder)

        for key, value in self._params["layers_2D"].items():
            update_params = {
                "activation": random.choice(["relu", "tanh", "swish"]),
                "dropout": random.randint(1, 3) * 0.1,
                "kernel": random.randint(3, 5),
                "stride": random.randint(1, 3),
                "padding": random.choice(["same", "valid"]),
                "units": random.choice([16, 32]),
            }
            self._params["layers_2D"].update({key: update_params})

        for key, value in self._params["layers_dense"].items():
            update_params = {
                "activation": random.choice(["relu", "tanh", "swish"]),
                "dropout": random.randint(1, 3) * 0.1,
                "units": random.choice([16, 32]),
            }
            self._params["layers_dense"].update({key: update_params})

        self._params.update(
            {"optimizer": random.choice(["rmsprop", "adam", "sgd", "adagrad"])}
        )
        self._params.update({"epochs": random.randint(10, 15)})


def rearrange_networks(networks):
    """
    combines the parameters of the best performing networks and
    adds offsprings to the population
    """
    offsprings = []
    for i in range(int((POPULATION - len(networks)) / 2)):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)
        offspring1 = Network()
        offspring2 = Network()

        for layer in offspring1._params["layers_2D"]:
            value = random.choice(
                [
                    random.choice(list(parent1._params["layers_2D"].values())),
                    random.choice(list(parent2._params["layers_2D"].values())),
                ]
            )
            offspring1._params["layers_2D"][layer].update(value)

        for layer in offspring2._params["layers_2D"]:
            value = random.choice(
                [
                    random.choice(list(parent1._params["layers_2D"].values())),
                    random.choice(list(parent2._params["layers_2D"].values())),
                ]
            )
            offspring2._params["layers_2D"][layer].update(value)

        for layer in offspring1._params["layers_dense"]:
            value = random.choice(
                [
                    random.choice(list(parent1._params["layers_dense"].values())),
                    random.choice(list(parent2._params["layers_dense"].values())),
                ]
            )
            offspring1._params["layers_dense"][layer].update(value)

        for layer in offspring2._params["layers_dense"]:
            value = random.choice(
                [
                    random.choice(list(parent1._params["layers_dense"].values())),
                    random.choice(list(parent2._params["layers_dense"].values())),
                ]
            )
            offspring2._params["layers_dense"][layer].update(value)
            # rearrage epochs and optimizer

            offspring1._params["epochs"], offspring2._params["epochs"] = (
                parent1._params["epochs"],
                parent2._params["epochs"],
            )
            offspring1._params["optimizer"], offspring2._params["optimizer"] = (
                parent2._params["optimizer"],
                parent1._params["optimizer"],
            )

        offsprings.append(offspring1)
        offsprings.append(offspring2)

    networks.extend(offsprings)
    return networks


def mutate_network(networks):
    """
    randomly renders individual parameters in 10% of all cases
    """
    for network in networks:
        for key, value in network._params["layers_2D"].items():
            if np.random.uniform(0, 1) <= 0.1:
                mut_dropout = (
                    network._params["layers_2D"][key].get("dropout")
                    + np.random.randint(-2, 2) * 0.1
                )
                network._params["layers_2D"][key].update({"dropout": mut_dropout})
            if np.random.uniform(0, 1) <= 0.1:
                mut_kernel = network._params["layers_2D"][key].get(
                    "kernel"
                ) + np.random.randint(-1, 1)
                network._params["layers_2D"][key].update({"kernel": mut_kernel})
            if np.random.uniform(0, 1) <= 0.1:
                mut_stride = network._params["layers_2D"][key].get(
                    "stride"
                ) + np.random.randint(-1, 1)
                network._params["layers_2D"][key].update({"stride": mut_stride})
            if np.random.uniform(0, 1) <= 0.1:
                mut_units = (
                    network._params["layers_2D"][key].get("units")
                    + np.random.randint(-2, 4) * 16
                )
                network._params["layers_2D"][key].update({"units": mut_units})

        for key, value in network._params["layers_dense"].items():
            if np.random.uniform(0, 1) <= 0.1:
                mut_dropout = (
                    network._params["layers_dense"][key].get("dropout")
                    + np.random.randint(-2, 2) * 0.1
                )
                network._params["layers_dense"][key].update({"dropout": mut_dropout})
            if np.random.uniform(0, 1) <= 0.1:
                mut_units = (
                    network._params["layers_dense"][key].get("units")
                    + np.random.randint(-2, 4) * 16
                )
                network._params["layers_dense"][key].update({"units": mut_units})

        if np.random.uniform(0, 1) <= 0.1:
            mut_epochs = network._params.get("epochs") + random.randint(-2, 4)
            network._params.update({"epochs": mut_epochs})

    return networks


(train_images, train_labels), (test_images, test_labels) = prepare_data(DATASET)


def optimizer():
    """
    main function, pipelines the networks through the optimization process
    logs best performing ones
    """
    networks = []
    for i in range(POPULATION):
        networks.append(Network())

    networks_accuracy = []
    best_network_accuracy = 0
    best_network = 0
    for generation in range(GENERATIONS):
        print(f"Generation number {generation}")
        total_accuracy = 0
        networks = assess_networks(networks)
        for network in networks:
            total_accuracy += network._accuracy

        networks = select_best_networks(networks)
        networks = rearrange_networks(networks)
        networks = mutate_network(networks)

        average_accuracy = total_accuracy / len(networks)
        networks_accuracy.append(average_accuracy)

        for network in networks:
            if network._accuracy > best_network_accuracy:
                best_network_accuracy = network._accuracy
                best_network = network
                logging.info(best_network.__dict__.items())
                print("current best accuracy: " + str(best_network_accuracy))

            if network._accuracy > THRESHOLD:
                print("Threshold met")
                print(network.__dict__.items())
                print("Best accuracy: {}".format(network._accuracy))
                logging.info(network.__dict__.items())
                exit(0)
        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info("-" * 80)
        logging.info("***Doing generation %d of %d***" % (generation + 1, GENERATIONS))

    return networks, networks_accuracy, best_network
    logging.info(
        "***Evolving %d generations with population %d***" % (GENERATIONS, population)
    )


if __name__ == "__main__":
    networks, networks_accuracy, best_network = optimizer()
