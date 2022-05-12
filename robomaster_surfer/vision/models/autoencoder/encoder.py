import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5,
                 bottleneck_activation=None, device='cpu'):
        """
        The function takes in the input size, hidden size, output size, dropout rate, bottleneck activation, and device.

        The function then initializes the super class, which is the nn.Module class.

        The function then sets the device, bottleneck activation, input size, and hidden size.

        The function then creates a linear layer with the input size and hidden size.

        :param input_size: the size of the input image
        :param hidden_size: the number of features in the hidden state h
        :param output_size: the number of classes in the dataset
        :param dropout_rate: the dropout rate for the encoder
        :param bottleneck_activation: The activation function to use for the bottleneck layer
        :param device: the device to run the model on, defaults to cpu (optional)
        """

        super(Encoder, self).__init__()
        self.device = device
        self.bottleneck_activation = bottleneck_activation
        self.input_size = input_size
        self.hidden_size = hidden_size
        if hidden_size == 0:
            hidden_size = 1
        self.linear0 = nn.Linear(input_size[0] * input_size[1], hidden_size)

    def forward(self, x):
        """
        It takes the input, flattens it, and then applies a linear transformation to it

        :param x: the input to the network
        :return: The output of the forward pass of the network.
        """
        if self.hidden_size == 0:
            return torch.zeros(x.shape[0], 1, device=self.device)
        x = x.view(-1, self.input_size[0] * self.input_size[1])
        if self.bottleneck_activation is None:
            x = self.linear0(x)
        else:
            x = self.bottleneck_activation(self.linear0(x))
        return x
