![alt text](https://travis-ci.org/mkoeppel/Optimizing_neural_nets.svg?branch=main)

# Optimizing_Neural_Nets 
Giving large enough datasets, artificial neural networks train themselves in order to become better at a given task. Yet, several hyperparameters need to be provided by the user beforehand and frequently need adjustment to obtain the desired output.
A possible alternative to such manual and user-defined adjustments are genetic algorithms, which allow a more automated and unbiased selection of best-performing hyperparameters:

Mimicking biologic evolutionary processes, this project adapts, recombines, and mutates hyperparameters from best-performing neural networks within a given population.

#### used tech:
![alt text](https://github.com/mkoeppel/Optimizing_neural_nets/blob/main/Tech_stack_opt_nets.jpeg)

The process starts with a random initialization of hyperparameters for a population of neural networks and tests their learning on the cifar10 dataset.

Subsequently, it selects the 20% best performing networks and generates offspring-networks be rearranging hyperparameters.
Two different ways are possible for rearranging parental layers and parameters for inclusion in offsprings:
- inheritance of complete layers: layers from the both parental networks are combined, so one from each into each offspring (resembling the shuffling of whole chromosomes in biology)
- ~~crossover of parameters: parental parameters are randomly chosen and included in the offsprings (resembling homologues recombination during meiosis)~~ this second option is currently inactive 

Afterwards (numeric) parameters have a 10% chance of being additionally rendered by a small value (resembling genetic point-mutations)

This performance-testing and rearranging of hyperparameters is done for > 20 rounds (generations), to select those network-settings that outperform the others by means of validation-accuracy.

This is the first version of an optimizing algorithm, keeping several hyperparameters set, while giving some flexibility for selective advantages to several others:

flexible hyperparameters are:
- random number of layers (both 2D- and dense- layers)
- number of units per layer
- dropout-rate per layer
- kernel- and stride size for the conv-layers
- padding with the conv-layers
- number of epochs: 10-15
- hidden layer activation: relu, tanh, swish
- optimizer: rmsprop, adam, sgd, adagrad

parameters that are fixed (for now):

- output activation function: softmax
- loss-function: categorical_crossentropy
- batch_size: 300


##### architecture of the best performing net after 20 generations:
![alt.text](https://github.com/mkoeppel/Optimizing_neural_nets/blob/main/best_performing_net.jpeg)


##### an example result from optimizing a neural net with 2 conv2D and 2 dense layers for 20 generations with a population size of 25:
![alt text](https://github.com/mkoeppel/Optimizing_neural_nets/blob/main/NeuralNet_opt_output.png)


### to do:
- ~~implement a random number of layers~~
- include other datasets for optimization like MNIST

This is a follow-up to the deep-learning week at the Data Science bootcamp at SPICED Academy and was inspired and guided by this blog-post:
https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164
