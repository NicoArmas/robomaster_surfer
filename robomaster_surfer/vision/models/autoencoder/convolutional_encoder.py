import torch
from torch import nn
import torch.nn.functional as F


class ConvolutionalEncoder(nn.Module):
    def __init__(self, dropout_rate=0.5):
        """
        We create a convolutional encoder with 4 convolutional layers, each with a dropout layer, and each with a different
        number of filters.

        The first layer has 16 filters, the second has 32, the third has 64, and the fourth has 128.

        The first three layers have a kernel size of 3, and the last layer has a kernel size of 6.

        The first three layers have a stride of 1, and the last layer has a stride of 2.

        The first three layers have a padding of 0, and the last layer has a padding of 1.

        The first three layers have a ReLU activation function, and the last layer has a tanh activation function.

        The first three layers have a max pooling layer, and the last layer does not.

        The first three layers have a dropout rate of 0.5, and the last layer does not

        :param dropout_rate: the dropout rate to use
        """
        super(ConvolutionalEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.conv0 = nn.Conv2d(1, 16, 3)
        self.conv1 = nn.Conv2d(16, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 6, stride=2, padding=1)

    def forward(self, x):
        """
        It takes an input, passes it through a series of convolutional layers, applies a dropout, and then returns the
        output

        :param x: the input to the network
        :return: The output of the last layer.
        """
        x = F.leaky_relu(self.conv0(x))
        x = self.dropout(F.leaky_relu(self.conv1(x)))
        x = self.dropout(F.leaky_relu(self.conv2(x)))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x
