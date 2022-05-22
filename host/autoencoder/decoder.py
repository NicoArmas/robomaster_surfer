import torch
from torch import nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5, device='cpu'):
        """
        The function takes in the input size, hidden size, output size, dropout rate, and device.

        The function then initializes the super class, which is the nn.Module class.

        The function then sets the output size and hidden size.

        The function then creates a linear layer with the hidden size and output size.

        :param input_size: the size of the input to the decoder
        :param hidden_size: the number of features in the hidden state h
        :param output_size: the size of the output image
        :param dropout_rate: the dropout rate for the decoder
        :param device: the device to run the model on, defaults to cpu (optional)
        """
        super(Decoder, self).__init__()
        self.device = device
        self.output_size = output_size
        if hidden_size == 0:
            hidden_size = 1
        self.linear0 = nn.Linear(hidden_size, output_size[0] * output_size[1])

    def forward(self, x):
        """
        It takes in an input, passes it through a linear layer, and then reshapes the output to the desired shape

        :param x: the input to the network
        :return: The output of the network is being returned.
        """
        x = torch.sigmoid(self.linear0(x))
        return x.view(-1, 1, self.output_size[0], self.output_size[1])
