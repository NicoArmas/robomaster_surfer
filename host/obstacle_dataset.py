import glob
import os

import cv2
import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from torch.utils.data.dataset import T_co


class ObstacleDataset(Dataset):

    def __init__(self, data_dir, transform=transforms.ToTensor()):
        self.samples = []
        self.data_dir = data_dir
        self.transform = transform
        self.targets = []
        self.labels = []
        targets = pd.read_csv(f'{os.getcwd()}/{data_dir}/targets.csv')["Label2"].to_numpy()

        for idx, target in enumerate(targets):
            target = torch.tensor(list(map(int, target.replace(' ', '').strip('(').strip(')').split(','))),
                                  dtype=torch.float)
            if len(target) == 1:
                continue
            self.labels.append(target.argmax().item())
            self.samples.append([f'img_{idx}.png', target])
            self.targets.append(target.type(torch.uint8).tolist())
        self.samples = np.array(self.samples, dtype=object)

    def insert(self, img, label, target, lane):
        list(self.samples).append((img, label))
        self.targets.append(target)
        self.labels.append(label)
        self.samples = np.array(self.samples, dtype=object)
        targets = pd.read_csv(f'{os.getcwd()}/{self.data_dir}/targets.csv')
        targets.loc[-1] = [img, str(target), lane]
        targets.index = targets.index + 1
        targets = targets.sort_values(by=['index'])
        targets.to_csv(f'{os.getcwd()}/{self.data_dir}/targets.csv', index=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]

        img = cv2.imread(f'{self.data_dir}/{img}')

        if self.transform is not None:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    dataset = ObstacleDataset('host/data/obstacle_avoidance')
    print(dataset.targets)
    print(len(dataset))
    print(dataset[0])
    print('The shape of tensor for 50th image in train dataset: ', dataset[-1][0].shape)
    print('The label for 50th image in train dataset: ', dataset[-1][1])
