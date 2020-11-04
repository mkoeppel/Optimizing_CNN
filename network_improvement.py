
import logging
import random
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

### some general parameters on which to optimize the network-training
dataset = cifar10
no_classes = 10
batch_size = 300
validation_split = 0.
epochs = 10
threshold = 0.9
population = 20
generations = 10
input_shape = (32, 32, 3)


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

class Network():
    def __init__(self):
        """
        parameters to be optimized:
        """
        self._layer2D = random.randint(0,4)
        self._units2D = random.choice([16, 32, 64, 128])
        self._stride = random.randint(1,4)
        self._dropout2D = random.uniform(0.0, 0.3)
        self._layerFull = random.randint(1,4)
        self._unitsFull = random.choice([32, 64, 128])
        self._dropoutFull = random.uniform(0.2, 0.5)

        """
        fixed parameters:
        """
        self._loss = 'categorical_crossentropy'
        self._kernel = 3
        self._hidden_activation = 'relu'
        self._output_activation = 'softmax'
        self._accuracy = 0.
        self._optimizer = 'adam'

    def create_model(self):
        """
        generates the actual models

        Params:
            input: list of parametes givven above
            output: model containing a Pooling and a flattening layer + output for no_classes
        """
        kernel = self._kernel
        stride = self._stride
        model = Sequential()
        model.add(InputLayer(input_shape = input_shape))
        if self._layer2D > 0:
            for layer in range(self._layer2D):
                model.add(Conv2D(self._units2D, kernel_size=(kernel, kernel), strides = (stride, stride), activation = self._hidden_activation))
                model.add(Dropout(self._dropout2D))
            model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
            model.add(Flatten())
        for layer in range(self._layerFull):
            model.add(Dense(self._unitsFull, activation = self._hidden_activation))
            model.add(Dropout(self._dropoutFull))
        model.add(Dense(no_classes, activation = self._output_activation))
        model.compile(loss = self._loss, optimizer = self._optimizer, metrics = ['accuracy'])

        return model

def init_networks(population):
    return [Network() for _ in range(population)]

def run_model(model):
    model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, verbose=1)
    return model

def assess_networks(networks):
    for network in networks:
        try:
            model = network.create_model()
            network = run_model(model)
            accuracy = network.evaluate(test_images, test_labels)[1]
            network._accuracy = accuracy
            print('Accuracy: {}'.format(network._accuracy))
        except:
            network._accuracy = 0
            print ('Build failed.')
    return networks

def select_best_networks(networks):
    networks = sorted(networks, key=lambda network: network._accuracy, reverse=True)
    networks = networks[:int(0.2 * len(networks))]

    return networks

def rearrange_networks(networks):
    offsprings = []
    for _ in range(int((population - len(networks)) / 2)):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)
        offspring1 = Network()
        offspring2 = Network()

        # rearange layers
        offspring1._layer2D = int((parent1._layer2D + parent2._layer2D)/2)
        offspring1._layerFull = int((parent1._layerFull + parent2._layerFull)/2)
        offspring2._layer2D = int((parent1._layer2D + parent2._layer2D)/2)
        offspring2._layerFull = int((parent1._layerFull + parent2._layerFull)/2)

        # rearrage units
        offspring1._units2D = int(parent1._units2D/2) + int(parent2._units2D)
        offspring1._unitsFull = int(parent1._unitsFull/2) + int(parent2._unitsFull)
        offspring2._units2D = int(parent1._units2D) + int(parent2._units2D/2)
        offspring2._unitsFull = int(parent1._unitsFull) + int(parent2._unitsFull/2)

        # rearrange kernel and strides
        offspring1._kernel = parent1._kernel
        offspring1._stride = parent2._stride
        offspring2._kernel = parent2._kernel
        offspring2._stride = parent1._stride

        # rearrange dropout
        offspring1._dropout2D = parent1._dropout2D
        offspring1._dropoutFull = parent2._dropoutFull
        offspring2._dropout2D = parent2._dropout2D
        offspring2._dropoutFull = parent1._dropoutFull

        offsprings.append(offspring1)
        offsprings.append(offspring2)

    networks.extend(offsprings)
    return networks

def mutate_network(networks):
    for network in networks:
        if np.random.uniform(0, 1) <= 0.1:
            network._units2D += np.random.randint(0,16)
        if np.random.uniform(0, 1) <= 0.1:
            network._unitsFull += np.random.randint(0,16)
        if np.random.uniform(0, 1) <= 0.1:
            network._kernel += np.random.randint(-1,1)
        if np.random.uniform(0, 1) <= 0.1:
            network._dropout2D += np.random.uniform(-0.2,0.2)
        if np.random.uniform(0, 1) <= 0.1:
            network._dropoutFull += np.random.uniform(-0.2,0.2)

    return networks

(train_images, train_labels), (test_images, test_labels) = prepare_data(dataset)

def optimizer():
    networks = init_networks(population)

    for generation in range(generations):
        print(f'Generation number {generation}')

        networks = assess_networks(networks)
        networks = select_best_networks(networks)
        networks = rearrange_networks(networks)
        networks = mutate_network(networks)

        for network in networks:
            if network._accuracy > threshold:
                print ('Threshold met')
                print (network.__dict__.items())
                print ('Best accuracy: {}'.format(network._accuracy))
                exit(0)
    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

if __name__ == '__main__':
    optimizer()
