import logging
import random
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation, Dropout, Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical

from helper_functions import prepare_data, assess_networks, select_best_networks


# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)
dataset = cifar10
no_classes = 10
batch_size = 300
validation_split = 0.2
threshold = 0.75
population = 20
generations = 20
input_shape = (32, 32, 3)



class Network():
    def __init__(self):
        """
        parameters to be optimized:
            Dense-layers: units, activation, dropout
        """

        self._layer_dense_1 = []
        self._layer_dense_2 = []
        self._layer_dense_3 = []

        self._layers_dense = [self._layer_dense_1, self._layer_dense_2, self._layer_dense_3]

        for layer in self._layers_dense:
            units = random.choice([32, 64, 128, 256])
            hidden_activation = random.choice(['relu', 'tanh', 'swish'])
            dropout = random.randint(2, 5)*0.1

            layer.extend([units, hidden_activation, dropout])

        self._optimizer = random.choice(['rmsprop', 'adam', 'sgd', 'adagrad'])
        self._epochs = random.randint(10, 15)
        """
        fixed parameters:
        """
        self._accuracy = 0
        self._loss = 'categorical_crossentropy'
        self._output_activation = 'softmax'
        self._genetic_function = 0


    def get_parameters(self):
        """
        provides a dictionary of prarameters fro each instantce of Network
        """
        parameters = {
          'layer_dense_1' : self._layer_dense_1,
          'layer_dense_2' : self._layer_dense_2,
          'layer_dense_3' : self._layer_dense_3,
          'loss' : self._loss,
          'output_activation' : self._output_activation,
          'optimizer' : self._optimizer,
          'epochs' : self._epochs
          }
        return parameters


def create_model(network):
    """
    generates the actual model

    Params:
      input: dictionary of parametes
      output: model containing a Pooling and a flattening layer + output for no_classes
    """

    parameters = network.get_parameters()

    layer_dense_1 = parameters['layer_dense_1']
    layer_dense_2 = parameters['layer_dense_2']
    layer_dense_3 = parameters['layer_dense_3']
    loss = parameters['loss']
    output_activation = parameters['output_activation']
    optimizer = parameters['optimizer']
    epochs = parameters['epochs']

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(layer_dense_1[0], activation=layer_dense_1[1]))
    model.add(Dropout(layer_dense_1[2]))
    model.add(Dense(layer_dense_2[0], activation=layer_dense_2[1]))
    model.add(Dropout(layer_dense_2[2]))
    model.add(Dense(layer_dense_3[0], activation=layer_dense_3[1]))
    model.add(Dropout(layer_dense_3[2]))
    model.add(Dense(no_classes, activation=output_activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, verbose=0)

    return model


def init_networks(population):
    """
    instantiate networks according to the number defined as population
    """
    return [Network() for _ in range(population)]


def rearrange_networks(networks):
    """
    combines the parameters of the best performing networks and
    adds offsprings to the population
    """
    offsprings = []
    for _ in range(int((population - len(networks)) / 2)):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)
        offspring1 = Network()
        offspring2 = Network()

        # select mutator type
        genetic_function = random.randint(1, 2)
        if genetic_function == 1:
            offspring1._genetic_function = 1
            offspring2._genetic_function = 1
            # rearrage layers
            offspring1._layer_dense_1, offspring1._layer_dense_2, offspring1._layer_dense_3 = parent1._layer_dense_1, parent2._layer_dense_2, parent1._layer_dense_3
            offspring2._layer_dense_1, offspring2._layer_dense_2, offspring2._layer_dense_3 = parent2._layer_dense_1, parent1._layer_dense_2, parent2._layer_dense_3,
        # rearrage epochs and optimizer
        offspring1._epochs, offspring2._epochs = parent1._epochs, parent2._epochs
        offspring1._optimizer, offspring2._optimizer = parent2._optimizer, parent1._optimizer

        else:
            offspring1._genetic_function = 2
            offspring2._genetic_function = 2
            #exchange parameters from the different parental layers
                for layer in offspring1._layers_dense:
                    for i in range(len(layer)):
                        layer[i] = random.choice([parent1._layer_dense_1[i], parent1._layer_dense_2[i],
                                                  parent2._layer_dense_1[i], parent2._layer_dense_2[i]])

                for layer in offspring2._layers_dense:
                    for i in range(len(layer)):
                        layer[i] = random.choice([parent1._layer_dense_1[i], parent1._layer_dense_2[i],
                                                  parent2._layer_dense_1[i], parent2._layer_dense_2[i]])
        # rearrage epochs and optimizer
        offspring1._epochs, offspring2._epochs = parent1._epochs, parent2._epochs
        offspring1._optimizer, offspring2._optimizer = parent2._optimizer, parent1._optimizer


        offsprings.append(offspring1)
        offsprings.append(offspring2)

    networks.extend(offsprings)
    return networks


def mutate_network(networks):
    """
    randomly renders individual parameters in 10% of all cases
    """
    for network in networks:

        for layer in network._layers_dense:
            if np.random.uniform(0, 1) <= 0.1:
                # mutate units
                layer[0] += np.random.randint(-2, 4)*16
            if np.random.uniform(0, 1) <= 0.1:
                # mutate dropout
                layer[2] += np.random.randint(-2, 2)*0.1
        if np.random.uniform(0, 1) <= 0.1:
            network._epochs += random.randint(-2, 2)
    return networks

train_images, train_labels, test_images, test_labels = prepare_data(dataset)

def optimizer():
    """
    main function, pipelines the networks through the optimization process
    logs best performing ones
    """
    networks = init_networks(population)
    networks_accuracy = []
    best_network_accuracy = 0
    best_network = 0
    for generation in range(generations):
        print(f'Generation number {generation}')
        total_accuracy = 0
        networks = assess_networks(networks)
        for network in networks:
            total_accuracy += network._accuracy

        networks = select_best_networks(networks)
        networks = rearrange_networks(networks)
        networks = mutate_network(networks)

        average_accuracy = total_accuracy/len(networks)
        networks_accuracy.append(average_accuracy)

        for network in networks:
            if network._accuracy > best_network_accuracy:
                best_network_accuracy = network._accuracy
                best_network = network
                logging.info(best_network.__dict__.items())
                print('current best accuracy: ' +str(best_network_accuracy))

            if network._accuracy > threshold:
                print('Threshold met')
                print(network.__dict__.items())
                print('Best accuracy: {}'.format(network._accuracy))
                logging.info(network.__dict__.items())
                exit(0)
        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy*100))
        logging.info('-'*80)
        logging.info("***Doing generation %d of %d***" %
                     (generation + 1, generations))


    return networks, networks_accuracy, best_network
    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

if __name__ == '__main__':
    networks, networks_accuracy, best_network = optimizer()
