import json
import random

from .exceptions import CanNotLoadNetworkConfiguration, CanNotSaveConfiguration
from .helpers import dot, neuron_output


class NeuralNetwork:

    NUM_HIDDEN_LAYERS = 5

    def __init__(self):
        self.model = []
        self.neural_network = []

    def feed_forward(self, input_vector):

        outputs = []

        for layer in self.neural_network:
            input_vector_bias = input_vector + [1]
            output = [neuron_output(neuron, input_vector_bias) for neuron in layer]
            outputs.append(output)
            input_vector = output

        return outputs

    def best_guess_for_vector(self, input_vector):
        results = self.feed_forward(input_vector)
        return results[1].index(max(results[1]))

    def test_data(self, data):
        """
        :param data: list of tuples: [ (value, [vector]), ...]
        """
        correct = 0
        for expected_value, vector_value in data:
            guess_index = self.best_guess_for_vector(vector_value)
            correct += int(bool(expected_value == self.model[guess_index][0]))
            print(expected_value, self.model[guess_index][0])
        print("Correct Guesses: {}/{}".format(correct, len(data)))

    def backpropagate(self, input_vector, train_data):
        """
        Update the weights of each of it neurons in its layers according to target
        outputs
        """
        hidden_outputs, outputs = self.feed_forward(input_vector)

        output_deltas = [output * (1-output) * (output - target) for output, target in zip(outputs, train_data)]
        for i, output_neuron in enumerate(self.neural_network[-1]):
            for j, hidden_output in enumerate(hidden_outputs + [1]):
                output_neuron[j] -= float(output_deltas[i] * hidden_output)

        hidden_deltas = [hidden_output * (1 - hidden_output) *
                         dot(output_deltas, [n[i] for n in self.neural_network[-1]])
                         for i, hidden_output in enumerate(hidden_outputs)]
        for i, hidden_neuron in enumerate(self.neural_network[0]):
            for j, input in enumerate(input_vector + [1]):
                hidden_neuron[j] -= float(hidden_deltas[i] * input)

    def train(self, inputs):
        """
        Train the network based on inputs which is a list of tuples:
              [ (expected_value, [vector value]), (), ... ]
        """
        train_data = [element[1] for element in inputs]
        target_data = [[1 if i == j else 0 for j in range(len(inputs))] for i in range(len(inputs))]

        for _ in range(10000):
            for input_vector, target_vector in zip(train_data, target_data):
                self.backpropagate(input_vector, target_vector)
        self.model = inputs

    def save_result(self, file_path):
        if not file_path:
            raise CanNotSaveConfiguration
        output = {'layers': self.neural_network, 'model': self.model}
        with open(file_path, 'w') as f:
            f.write(json.dumps(output))

    def load_configuration_from_file(self, file_path):
        content = ''
        with open(file_path, 'r') as f:
            content = json.loads(f.read())
        try:
            self.neural_network = content['layers']
            self.model = content['model']
        except ValueError:
            raise CanNotLoadNetworkConfiguration

    def initiate_network_configuration(self, input_size, output_size):
        hidden_layer = [[random.random() for _ in range(input_size + 1)] for _ in range(self.NUM_HIDDEN_LAYERS)]
        output_layer = [[random.random() for _ in range(self.NUM_HIDDEN_LAYERS + 1)] for _ in range(output_size)]

        self.neural_network = [hidden_layer, output_layer]
