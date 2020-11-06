
import logging
import random
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, ZeroPadding2D
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
generations = 50
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
        self._units2D_1 = random.choice([16, 32, 64, 128])
        self._dropout2D_1 = random.uniform(0.0, 0.3)
        self._kernel_1 = random.randint(3,5)
        self._stride_1 = random.randint(1,3)
        self._padding_1 =random.choice(['same', 'valid'])
        self._units2D_2 = random.choice([16, 32, 64, 128])
        self._dropout2D_2 = random.uniform(0.0, 0.3)
        self._kernel_2 = random.randint(3,5)
        self._stride_2 = random.randint(1,3)
        self._padding_2 =random.choice(['same', 'valid'])

        self._unitsFull_1 = random.choice([32, 64, 128])
        self._dropoutFull_1 = random.uniform(0.2, 0.5)
        self._unitsFull_2 = random.choice([32, 64, 128])
        self._dropoutFull_2 = random.uniform(0.2, 0.5)


        """
        fixed parameters:
        """
        self._loss = 'categorical_crossentropy'
        self._hidden_activation = 'relu'
        self._output_activation = 'softmax'
        self._optimizer = 'adam'

        self._accuracy = 0

    def create_model(self):
        """
        generates the actual models

        Params:
            input: list of parametes given above
            output: model containing a Pooling and a flattening layer + output for no_classes
        """
        kernel_1 = (self._kernel_1, self._kernel_1)
        strides_1 = (self._stride_1, self._stride_1)
        kernel_2 = (self._kernel_2, self._kernel_2)
        strides_2 = (self._stride_2, self._stride_2)

        model = Sequential()
        model.add(InputLayer(input_shape = input_shape))
        model.add(Conv2D(self._units2D_1, kernel_size= kernel_1, strides = strides_1, activation = self._hidden_activation))
        model.add(Dropout(self._dropout2D_1))
        model.add(Conv2D(self._units2D_2, kernel_size=kernel_2,strides = strides_1, activation = self._hidden_activation))
        model.add(Dropout(self._dropout2D_2))
        model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
        model.add(Flatten())
        model.add(Dense(self._unitsFull_1, activation = self._hidden_activation))
        model.add(Dropout(self._dropoutFull_1))
        model.add(Dense(self._unitsFull_2, activation = self._hidden_activation))
        model.add(Dropout(self._dropoutFull_2))
        model.add(Dense(no_classes, activation = self._output_activation))
        model.compile(loss = self._loss, optimizer = self._optimizer, metrics = ['accuracy'])
        model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, verbose=1)

        return model

def init_networks(population):
    return [Network() for _ in range(population)]

def assess_networks(networks):
    for network in networks:
        try:
            model = create_model(network)
            accuracy = model.evaluate(test_images, test_labels, verbose=0)[1]
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

        # rearrage units
        offspring1._units2D_1 = int(parent1._units2D_1/2) + int(parent2._units2D_1)
        offspring1._unitsFull_1 = int(parent1._unitsFull_1/2) + int(parent2._unitsFull_1)
        offspring2._units2D_1 = int(parent1._units2D_1) + int(parent2._units2D_1/2)
        offspring2._unitsFull_1 = int(parent1._unitsFull_1) + int(parent2._unitsFull_1/2)

        offspring1._units2D_2 = int(parent1._units2D_2/2) + int(parent2._units2D_2)
        offspring1._unitsFull_2 = int(parent1._unitsFull_2/2) + int(parent2._unitsFull_2)
        offspring2._units2D_2 = int(parent1._units2D_2) + int(parent2._units2D_2/2)
        offspring2._unitsFull_2 = int(parent1._unitsFull_2) + int(parent2._unitsFull_2/2)

        # rearrange kernel and strides
        offspring1._kernel_1, offspring1._kernel_2 = parent1._kernel_1, parent2._kernel_2
        offspring1._stride_1, offspring2._stride_2 = parent1._stride_1, parent2._stride_2
        offspring2._kernel_1, offspring2._kernel_2 = parent1._kernel_2, parent2._kernel_1
        offspring2._stride_1, offspring2._stride_2 = parent1._stride_2, parent2._stride_1

        # rearrange dropout
        offspring1._dropout2D_1, offspring1._dropout2D_2  = parent1._dropout2D_1, parent2._dropout2D_2
        offspring1._dropoutFull_1, offspring1._dropoutFull_2  = parent1._dropoutFull_1, parent2._dropoutFull_2
        offspring2._dropout2D_1, offspring2._dropout2D_2 = parent1._dropout2D_2, parent2._dropout2D_1
        offspring2._dropoutFull_1, offspring2._dropoutFull_2 = parent1._dropoutFull_2, parent2._dropoutFull_1

        offsprings.append(offspring1)
        offsprings.append(offspring2)

    networks.extend(offsprings)
    return networks

def mutate_network(networks):
    for network in networks:
        if np.random.uniform(0, 1) <= 0.1:
            network._units2D_1 += np.random.randint(0,32)
        if np.random.uniform(0, 1) <= 0.1:
            network._unitsFull_1 += np.random.randint(0,32)
        if np.random.uniform(0, 1) <= 0.1:
            network._dropout2D_1 += np.random.uniform(-0.2,0.2)
        if np.random.uniform(0, 1) <= 0.1:
            network._dropoutFull_1 += np.random.uniform(-0.2,0.2)

    return networks

(train_images, train_labels), (test_images, test_labels) = prepare_data(dataset)

def optimizer():
    networks = init_networks(population)
    networks_accuracy = []
    best_network_accuracy = 0
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
              print('current best accuracy: ' +str(best_network_accuracy))

          if network._accuracy > threshold:
              print ('Threshold met')
              print (network.__dict__.items())
              print ('Best accuracy: {}'.format(network._accuracy))
              logging.info(network.__dict__.items())
              exit(0)
        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)
        logging.info("***Doing generation %d of %d***" %
          (generation + 1, generations))


    return networks, networks_accuracy
    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

if __name__ == '__main__':
  networks, networks_accuracy = optimizer()
