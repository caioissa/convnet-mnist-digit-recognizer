import os

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_path = 'model/model.pytorch'
        self.dense = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(1),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        return self.dense(x)

    def load(self):
        if os.path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path))

    def save(self):
        if not os.path.exists('model'):
            os.makedirs('model')
        torch.save(self.state_dict(), self.model_path)
