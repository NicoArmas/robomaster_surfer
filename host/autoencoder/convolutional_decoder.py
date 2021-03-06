import torch
from torch import nn
import torch.nn.functional as F


class ConvolutionalDecoder(nn.Module):
    def __init__(self, output_size, dropout_rate=0.5):
        """
        The function takes in a tensor of size 128 and outputs a tensor of size 1

        :param output_size: The size of the output image
        :param dropout_rate: The dropout rate to use
        """
        super(ConvolutionalDecoder, self).__init__()
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout_rate)
        self.deconv0 = nn.ConvTranspose2d(128, 64, 6, stride=2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3)
        self.deconv3 = nn.ConvTranspose2d(16, 3, 3)

    def forward(self, x):
        """
        We take the input, reshape it to a 4x4 image, then apply a series of deconvolutional layers to upsample it to the
        desired output size

        :param x: the input to the network
        :return: The output of the last layer, which is the reconstructed image.
        """
        x = x.view(-1, 128, 29, 29)
        x = F.leaky_relu(self.deconv0(x))
        x = self.dropout(F.leaky_relu(self.deconv1(x)))
        x = self.dropout(F.leaky_relu(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))
        return x[:, :, :self.output_size[0], :self.output_size[1]]
