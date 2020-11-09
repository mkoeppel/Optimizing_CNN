# Optimizing_Neural-Nets
Mimicking biologic evolutionary processes, this project adapts, recombines, and mutates hyperparameters from best-performing neural networks within a given population.

It performs an initial random initialization of hyperparameters for a population of neural networks and tests their learning on the cifar10 dataset.

Subsequently, it selects the 20% best performing networks and generates offspring-networks be rearranging hyperparameters.
Two different ways are possible for rearranging parental layers and parameters for inclusion in offsprings:
- inheritance of complete layers: layers from the both parental network are combined, so one from each into each offspring (resembling the shuffling of whole chromosomes in biology)
- crossover of parameters: parental parameters are randomly chosen and included in the offsprings (resembling the biological homologues recombination)
Afterwards (numeric) parameters have a 10% chance of being additionally rendered by a small value (resembling genetic point-mutations)

This performance-testing and rearranging of hyperparameters is done for > 20 rounds (generations), to select those network-settings that outperform the others by means of validation-accuracy.

This is the first version of an optimizing algorithm, keeping several hyperparameters set, while giving some flexibility for selective advantages to several others:

flexible hyperparameters are:
- number of units per layer
- dropout-rate per layer
- kernel- and stride size for the conv-layers
- padding with the conv-layers
- number of epochs: 10-15
- hidden layers activation: relu, tanh, swish
- optimizer: rmsprop, adam, sgd, adagrad

parameters that are fixed (for now):
- number of conv-layers: 2
- number of dense layers: 2 (4 in network_improvement_DenseOnly.py)
- output activation-function: softmax
- loss-function: categorical_crossentropy
- batch_size: 300

This is a follow-up to the deep-learning week at the Data Science bootcamp at SPICED Academy
