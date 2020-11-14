"""
tests for Optimizing_neural_nets
"""

import pytest
import random
import copy
import numpy as np

from network_improvement import Network, init_networks, rearrange_networks, mutate_network

def test_Network_class():
    network = Network()
    parameters = network.get_parameters()
    assert isinstance(parameters, dict)

def test_parameter_datatype():
    network = Network()
    parameters = network.get_parameters()
    assert isinstance(parameters['layers_2D'], list)
    assert isinstance(parameters['optimizer'], str)

def test_rearrangement():
    population = 4
    parents = init_networks(2)
    networks = rearrange_networks(parents)
    offspring_parameters = networks[2].get_parameters()
    parents_parameters = parents[0].get_parameters()
    assert parents_parameters.items() != offspring_parameters.items()

def test_mutation():
    networks = init_networks(2)
    networks_mut = networks.copy()
    network_parameters = networks[0].get_parameters()
    for i in range(10):
        networks_mut = mutate_network(networks_mut)
    mut_parameters = networks_mut[0].get_parameters()
    assert network_parameters.items() != mut_parameters.items()



def test_sanity():
    number = random.randint(1,10)
    assert type(number) == int
