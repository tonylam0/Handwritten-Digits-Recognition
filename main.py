import numpy as np
import random


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)  # Number of layers in neural network
        self.sizes = sizes

        '''
        Creates a list of column vectors that holds the biases 
        for each neuron in the layers after the input layer.
        The input layer does not have biases in a neural network.
        We use np.random.randn because if the range is too extreme
        network will become stuck, as the sigmond function will
        will be too far from 0 to shift from 0 to 1 or vise versa.
        '''
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        '''
        Makes a list of matrices that holds the weights for each neuron.
        Each column in each matrix represent the weights of 1 input neuron.
        '''
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]

    # Returns a vector of outputs given a vector of inputs
    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    '''
    Does the initial stages of creating a neural network (i.e.
    shuffles the training data for prepare for mini batches,
    creates mini-batches, prints accuracy of the network).
    '''
    def initiate_SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        '''
        training_data is a list of tuples "(x, y)" representing training inputs
        and their expected outputs with x being an array of the pixels
        a training image, and y being a vector of the expected output.
        '''
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for i in range(epochs):
            random.shuffle(training_data)

            # Creates subsets within the dataset with size = mini_batch_size
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print("epoch {0}: {1} / {2}".format(
                    i, self.evaluate(test_data), n_test))
            else:
                print("epoch {0}: complete".format())

    def update_mini_batch(self, mini_batch, eta):
        pass


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


