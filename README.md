# Optimizing_CNNs
Mimicking biologic evolutionary processes, this script adapts, recombines, and mutates hyperparameters from best-performing CNNs.

It performs an inital random initialization of hyperparamters for a population of convolutional neural networks and tests their learning on the cifar10 dataset.
Subsequently, it selects the 20% best performing networks and generates offspring-networks be rearrenging hyperparameters and chaning them randomly in 10% of all cases. This performance-testing and rearranging of hyperparameters is done for > 50 rounds, to select for those network-settings that outperform the others be means of validation-accuracy. 

This is a follow-up to the deep-learning week at the Data Science bootcamp at SPICED Academy
