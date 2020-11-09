import logging
import random
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation, Dropout, Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical

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


def prepare_data(dataset):
    """
    loads a given dataset, splits into X (images) and y (labels),
    selects only the first 10000 images, splits into train and test-set

    Params:
        input: a keras dataset
        output: x_train, y_train, x_test, y_test
    """
    (xtrain, ytrain), (xtest, ytest) = dataset.load_data()
    train_images = xtrain / 255
    test_images = xtest / 255
    train_labels = to_categorical(ytrain, 10)
    test_labels = to_categorical(ytest, 10)

    return train_images, train_labels, test_images, test_labels

class Network():
    def __init__(self):
        """
        parameters to be optimized:
            Dense-layers: units, activation, dropout
        """

        self._layerDense_1 = []
        self._layerDense_2 = []
        self._layerDense_3 = []

        self._layersDense = [self._layerDense_1, self._layerDense_2, self._layerDense_3]

        for layer in self._layersDense:
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
          'layerDense_1' : self._layerDense_1,
          'layerDense_2' : self._layerDense_2,
          'layerDense_3' : self._layerDense_3,
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

    layerDense_1 = parameters['layerDense_1']
    layerDense_2 = parameters['layerDense_2']
    layerDense_3 = parameters['layerDense_3']
    loss = parameters['loss']
    output_activation = parameters['output_activation']
    optimizer = parameters['optimizer']
    epochs = parameters['epochs']

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(layerDense_1[0], activation=layerDense_1[1]))
    model.add(Dropout(layerDense_1[2]))
    model.add(Dense(layerDense_2[0], activation=layerDense_2[1]))
    model.add(Dropout(layerDense_2[2]))
    model.add(Dense(layerDense_3[0], activation=layerDense_3[1]))
    model.add(Dropout(layerDense_3[2]))
    model.add(Dense(no_classes, activation=output_activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, verbose=0)

    return model


def init_networks(population):
    """
    instantiate networks according to the number defined as population
    """
    return [Network() for _ in range(population)]

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
            print('Accuracy: {}'.format(network._accuracy))

        except:
            network._accuracy = 0
            print ('Build failed.')
    return networks


def select_best_networks(networks):
    """
    sorts models by best performance and keeps only top 20%
    """
    networks = sorted(networks, key=lambda network: network._accuracy, reverse=True)
    networks = networks[:int(0.2 * len(networks))]

    return networks

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
            offspring1._genetic_function == 1
            offspring2._genetic_function == 1
            # rearrage layers
            offspring1._layerDense_1, offspring1._layerDense_2, offspring1._layerDense_3 = parent1._layerDense_1, parent2._layerDense_2, parent1._layerDense_3
            offspring2._layerDense_1, offspring2._layerDense_2, offspring2._layerDense_3 = parent2._layerDense_1, parent1._layerDense_2, parent2._layerDense_3,

        else:
            offspring1._genetic_function == 2
            offspring2._genetic_function == 2
            #exchange parameters from the different parental layers
            for layer in offspring1._layersDense:
                layer[0] = random.choice([parent1._layerDense_1[0], parent1._layerDense_2[0], parent1._layerDense_3[0],
                                        parent2._layerDense_1[0], parent2._layerDense_2[0], parent2._layerDense_3[0]])
                layer[1] = random.choice([parent1._layerDense_1[1], parent1._layerDense_2[1], parent1._layerDense_3[1],
                                        parent2._layerDense_1[1], parent2._layerDense_2[1], parent2._layerDense_3[1]])
                layer[2] = random.choice([parent1._layerDense_1[2], parent1._layerDense_2[2], parent1._layerDense_3[2],
                                        parent2._layerDense_1[2], parent2._layerDense_2[2], parent2._layerDense_3[2]])


        offsprings.append(offspring1)
        offsprings.append(offspring2)

    networks.extend(offsprings)
    return networks


def mutate_network(networks):
    """
    randomly renders individual parameters in 10% of all cases
    """
    for network in networks:

        for layer in network._layersDense:
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
              logging.info('current best network accurancy: ' +str(best_network_accuracy))
              logging.info(best_network.__dict__.items())
              print('current best accuracy: ' +str(best_network_accuracy))

          if network._accuracy > threshold:
              print ('Threshold met')
              print (network.__dict__.items())
              print ('Best accuracy: {}'.format(network._accuracy))
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
