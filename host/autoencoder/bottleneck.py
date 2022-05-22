import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, input_size, bottleneck_size, bottleneck_activation):
        """
        The function takes in the input size, the bottleneck size, and the bottleneck activation function.

        The function then creates a linear layer with the input size and the bottleneck size.

        The function then creates a linear layer with the bottleneck size and the input size.

        The function then returns the input size, the output size, the bottleneck size, and the bottleneck activation
        function.

        :param input_size: the size of the input to the bottleneck layer
        :param bottleneck_size: The size of the bottleneck layer
        :param bottleneck_activation: The activation function to use for the bottleneck layer
        """
        super(Bottleneck, self).__init__()
        self.input = int(np.prod(input_size))
        self.output = input_size
        self.bottleneck_size = bottleneck_size
        self.bottleneck_activation = bottleneck_activation
        self.fc1 = nn.Linear(self.input, bottleneck_size)
        self.fc2 = nn.Linear(bottleneck_size, self.input)

    def forward(self, x):
        """
        The function takes in an input, reshapes it, passes it through a linear layer, reshapes it again, and returns the
        output

        :param x: the input to the model
        :return: The output of the forward pass, and the encoded representation of the input.
        """
        x = x.view(-1, self.input)
        if self.bottleneck_activation is not None:
            encoded = self.bottleneck_activation(self.fc1(x))
        else:
            encoded = self.fc1(x)
        x = self.fc2(encoded)
        x = x.view(-1, self.output[0], self.output[1], self.output[2])
        return x, encoded
