import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torchinfo

from host.autoencoder import Autoencoder


class ObstacleAvoidance(nn.Module):
    def __init__(self, dropout_rate=0.3):
        """
        We take in an image, pass it through a series of convolutional layers, and then pass the output of the last
        convolutional layer through a linear layer to get a 3-dimensional output

        :param dropout_rate: the rate at which we dropout neurons in the network
        """
        super(ObstacleAvoidance, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.conv0 = nn.Conv2d(3, 16, 3)
        self.conv1 = nn.Conv2d(16, 32, 3)
        self.mp0 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.mp1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 5, stride=2, padding=1)
        self.mp2 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, 5, stride=2, padding=1)

        self.lin0 = nn.Linear(256, 3)

    def forward(self, x):
        """
        We take an input image, pass it through a series of convolutional layers, max pooling layers, and fully connected
        layers, and return a vector of probabilities that tells the robot which lane to move in

        :param x: the input to the network
        :return: The output of the last layer of the network.
        """
        x = F.leaky_relu(self.conv0(x))
        x = self.dropout(F.leaky_relu(self.conv1(x)))
        x = self.mp0(x)
        x = self.dropout(F.leaky_relu(self.conv2(x)))
        x = self.mp1(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.mp2(x)
        x = F.leaky_relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = self.lin0(x)
        return x

    def get_model_info(self):
        """
        It takes a model and returns a summary of the model
        """
        return torchinfo.summary(self, input_size=(1, 3, 128, 128), verbose=2)


if __name__ == '__main__':
    a = ObstacleAvoidance().to('cuda')
    b = transforms.ToTensor()

    frame = torch.ones(size=(1, 3, 128, 128)).to('cuda')
    print(a(frame))
    model_info = a.get_model_info()
    with open('model_info.txt', 'w') as f:
        f.write(str(model_info))
