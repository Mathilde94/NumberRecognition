import getopt
import sys

from data.constants import INPUT_NUMBERS, OUTPUT_FILE , VERIFY_NUMBERS
from neural_network.model import NeuralNetwork


def load(file=None, inputs=INPUT_NUMBERS):
    """
    Trains a neural network and writes its layer configuration and model
    in the file
    :param file: File where the Neural Network layers configurations will be stored
    :param inputs: Inputs to build the neural network on
    """
    if not file or not inputs:
        raise Exception("Can not load the file or inputs empty")

    network = NeuralNetwork()
    network.initiate_network_configuration(len(inputs[0][1]), len(inputs))
    network.train(inputs)
    network.save_result(file)


def verify(file=OUTPUT_FILE, inputs=VERIFY_NUMBERS):
    """
    Verifies a simple input from a neural network where the configuration
    is written in the file parameter.
    :param file: File where the Neural Network layers configurations are
    :param inputs: Inputs to verify against this Neural Network
    """
    network2 = NeuralNetwork()
    network2.load_configuration_from_file(OUTPUT_FILE)
    network2.test_data(inputs)

MAP_ACTIONS = {'load': load, 'verify': verify}


def show_use():
    print("Use: python main.py <load|verify> -i <input_file>")
    sys.exit(1)


def main(argv):
    inputfile = OUTPUT_FILE

    try:
        if len(argv) < 1:
            show_use()
        action = argv[0]
        if action not in MAP_ACTIONS.keys():
            show_use()

        opts, args = getopt.getopt(argv[1:], "hi:", ["input-file="])
        for opt, arg in opts:
            if opt == '-h':
                show_use()
            if opt == '-i':
                inputfile = arg

        MAP_ACTIONS[action](file=inputfile)

    except getopt.GetoptError:
        show_use()

if __name__ == "__main__":
    main(sys.argv[1:])

