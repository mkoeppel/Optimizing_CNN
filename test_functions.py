"""
tests for Optimizing_neural_nets
"""

import pytest
import random
import copy
import numpy as np

from network_improvement import Network, rearrange_networks, mutate_network

def test_Network_class():
    network = Network()
    parameters = network._params
    assert isinstance(parameters, dict)

def test_parameter_datatype():
    network = Network()
    parameters = network._params
    assert isinstance(parameters['layers_2D'], dict)
    assert isinstance(parameters['optimizer'], str)

def test_rearrangement():
    POPULATION = 4
    parents = []
    networks = []
    for i in range(POPULATION):
        parents.append(Network())
    networks = rearrange_networks(parents)
    offspring_parameters = networks[2]._params
    parents_parameters = parents[0]._params
    assert parents_parameters.items() != offspring_parameters.items()



def test_sanity():
    number = random.randint(1,10)
    assert type(number) == int
