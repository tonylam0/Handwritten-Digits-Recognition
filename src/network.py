import numpy as np
import random


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)  # Number of layers in neural network
        self.sizes = sizes

        # Creates a list of column vectors that holds the biases 
        # for each neuron in the layers after the input layer.
        # The input layer does not have biases in a neural network.
        # We use np.random.randn because if the range is too extreme
        # network will become stuck, as the sigmond function will
        # will be too far from 0 to shift from 0 to 1 or vise versa.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # Makes a list of matrices that holds the weights for each neuron.
        # Each column in each matrix represent the weights of 1 input neuron.
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]

    # Returns a vector of the output activations given the activations of the 
    # input layer. It is fed forward layer by layer.
    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def initiate_SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        '''
        Does the initial stages of creating a neural network (i.e.
        shuffles the training data for prepare for mini batches,
        creates mini-batches, prints accuracy of the network). 
        training_data is a list of tuples "(x, y)" representing training inputs
        and their expected outputs with x being an array of the pixels
        a training image, and y being a vector of the expected output.
        '''
        if test_data:
            n_test = len(test_data)  # Number of test images
        n = len(training_data)  # Number of training images

        for i in range(epochs):
            random.shuffle(training_data)

            # Creates subsets within the dataset with size = mini_batch_size
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            # Used to track the accuracy of the neural network
            if test_data:
                print("epoch {0}: {1} / {2}".format(
                    i, self.evaluate(test_data), n_test))
            else:
                print("epoch {0}: complete".format())

    def update_mini_batch(self, mini_batch, eta):
        # Creates an empty matrix that will hold the partial
        # derivatives of the parameters. The matrix's dimensions
        # are equal to the dimensions defined in the initization
        # function of the class.
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # Returns you a tuple of (partial_b, partial_w), which
            # are layer by layer lists of arrays with the partials
            # of cost function C.
            partial_b, partial_w = self.backprop(x, y)

            # Updates the list of arrays with the new partial derivatives
            # layer by layer. Each element represents the rate of change
            # of cost with respect to the parameter.
            gradient_b = [gb + pb for gb, pb in zip(gradient_b, partial_b)]
            gradient_w = [gw + pw for gw, pw in zip(gradient_w, partial_w)]

        # Updates each bias/weight with the average of the gradient vectors
        self.biases = [curr_bias - eta/len(mini_batch) * gb
                    for curr_bias, gb in zip(self.biases, gradient_b)]
        self.weights = [curr_weight - eta/len(mini_batch) * gw
                        for curr_weight, gw in zip(self.weights, gradient_w)]

    def backprop(self, x, y):
        '''
        Calculates the gradient for cost C in terms of bias and weight
        layer by layer.
        '''
        gradient_b = [np.zeros(b.shape) for b in self.biases] 
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        a = x  # array of input pixels acts as the activations of input layer
        activations = [a]  # Stores activations, layer by layer
        z_list = []  # Stores weighted inputs z, layer by layer

        # Computes the activation and weighted input layer by layer
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, a) + bias  # Dot products the weight matrix and activation vector, then add bias vector
            a = sigmoid(z)  # Converts weighted input vector into activation vector
            z_list.append(z)  # Appends layer of weighted inputs for later calculating error
            activations.append(a)  # Appends layer of activations

        # Based off BP1 equation
        # Returns you the error of the output layer
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(z_list[-1])

        # Based off BP3 equation
        # Partial C/partial b of output layer
        gradient_b[-1] = delta

        # Based off BP4 equation
        # Requires transpose due to matrix multiplication rules
        # Partial C/partial w of output layer
        gradient_w[-1] = np.dot(delta, activations[-2].transpose())

        # Based off BP2 equation
        # Backwards pass the error
        # l represents the current layer of calculation
        # l starts at 2 because it's the layer before the output layer
        # As l increases, the layers go down/backwards
        for l in range(2, self.num_layers):
            z = z_list[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            # partial C/ partial b or w of non-output layers
            gradient_b[-l] = delta
            gradient_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (gradient_b, gradient_w)

    def cost_derivative(self, output_activations, y):
        '''
        Returns a vector of partial derivatives of cost C
        with respect to a, which is the output layer's activations 
        '''
        return (output_activations - y)
    
    def evaluate(self, test_data):
        '''
        Returns the total number of test inputs for which the neural
        network outputs the correct result.
        '''
        # Appends the neural network's output number/guess.
        # The neural network's output is the neuron with 
        # the highest activation out of the output layer's vector.
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))