from torch import nn
import numpy as np


def relu_forward(inputs):
    """

    :param inputs: inputs to the rectified linear layer
    :return: the output after applying the rectified linear activation function
    """
    pass


def relu_backward(grad, inputs):
    """

    :param grad: the backpropagated gradients
    :param inputs: the inputs that were given to the rectified linear layer
    :return: the gradient with respect to the inputs
    """
    pass


def linear_forward(inputs, weights, biases):
    """

    :param inputs: inputs to the linear layer
    :param weights: the weight parameters
    :param biases: the bias parameters
    :return: the output after applying the linear transformation
    """
    pass


def linear_backward(grad, inputs, weights):
    """

    :param grad: the backpropagated gradient
    :param inputs: the inputs that were given to the linear layer
    :param weights: the weight parameters
    :return: the gradient with respect to the weights, biases, and inputs
    """
    pass


def get_nnet_model() -> nn.Module:
    """ Get the neural network model

    @return: neural network model
    """
    pass


def train_nnet(nnet: nn.Module, states_nnet: np.ndarray, outputs: np.ndarray):
    """

    :param nnet: The neural network
    :param states_nnet: states (inputs)
    :param outputs: the outputs
    :return: None
    """
    pass
