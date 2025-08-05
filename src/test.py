import network as network
import mnist_loader


# Validation data isn't required in this form of neural network
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# 784 is the number of pixels in a 28x28 image within the MNIST data set.
# The number of hidden layers is adjustable with more layers being 
# more accurate but also slower.
# 10 is for the numbers 0-9.
net = network.Network([784, 30, 10])

net.initiate_SGD(training_data, 30, 10, 3.0, test_data=test_data)