# Optimizing_CNNs
Mimicking biologic evolutionary processes, this script adapts, recombines, and mutates hyperparameters from best-performing CNNs.

It performs an inital random initialization of hyperparamters for a population of convolutional neural networks and tests their learning on the cifar10 dataset.
Subsequently, it selects the 20% best performing networks and generates offspring-networks be rearranging hyperparameters and changing them randomly in 10% of all cases. This performance-testing and rearranging of hyperparameters is done for > 50 rounds, to select for those network-settings that outperform the others by means of validation-accuracy.

This is the first version of an optimizing algorithm, keeping several hyperparameters set, while given some flexibility for selection advantage to several others:

flexible hyperparameters are:
- number of units per layer
- dropout-rate per layer
- kernel- and stride size for the conv-layers
- padding with the conv-layers

parameters that are fixed (for now):
- number of conv-layers: 2
- number of dense layers: 2
- number of epochs: 10
- activation-function: relu for hidden layers, softmax for output layer
- loss-function: categorical_crossentropy
- optimizer: adam
- batch_size: 300

This is a follow-up to the deep-learning week at the Data Science bootcamp at SPICED Academy
