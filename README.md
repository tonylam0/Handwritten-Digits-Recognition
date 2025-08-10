# Handwritten Digits Recognition

## About this Project
This program implements a feedforward neural network in order to recognize handwritten digits using Python and NumPy.

The neural network learns through processing MNIST's 60,000 training images from the dataset and adjusting the weights and bias of the neural network through mini-batch gradient descent and backpropagation. After each epoch, the neural network is then evaluated on accuracy with MNIST's 10,000 test images.

The code is based on the concepts and explanations from the book ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com) by Michael Nielson.

## Features
- Feedforward neural network with customizable layer sizes
- Mini-batch gradient descent training with mini-batches
- Error minimization using backpropagation