import math


def dot(a, b):
    return sum([i*j for i, j in zip(a, b)])


def sigmoid(t):
    return 1 / float(1 + math.exp(-t))


def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))
