import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from host.autoencoder import Autoencoder


class ObstacleAvoidance(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ObstacleAvoidance, self).__init__()
        # self.autoencoder = Autoencoder((128, 128), 32, (128, 128),
        #                                convolutional=True, dropout_rate=0,
        #                                bottleneck_activation=None).to('cuda')
        # self.autoencoder.load_state_dict(torch.load('model.pt'))
        self.dropout = nn.Dropout(dropout_rate)
        self.conv0 = nn.Conv2d(3, 16, 3)
        self.conv1 = nn.Conv2d(16, 32, 3)
        self.mp0 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.mp1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 5, stride=2, padding=1)
        self.mp2 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, 5, stride=2, padding=1)

        self.lin0 = nn.Linear(256, 8)

    def forward(self, x):
        # self.autoencoder.eval()
        # _, ret = self.autoencoder(x)
        # x = torch.clip(x - ret, 0, 1)
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


if __name__ == '__main__':
    a = ObstacleAvoidance().to('cuda')
    b = transforms.ToTensor()

    frame = torch.ones(size=(1, 3, 128, 128)).to('cuda')
    print(a(frame))
