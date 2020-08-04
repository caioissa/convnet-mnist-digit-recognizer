import random

import torch
import matplotlib.pyplot as plt
import numpy as np

from get_data import get_data
from model import Model
from train import check_for_GPU

dataloaders = get_data()

device = check_for_GPU()

model = Model().to(device)
model.load()

sample_size = 10

temp_dataloader = get_data()['val']
image_batch = random.choice([e[0] for e in list(iter(temp_dataloader))])

x = image_batch.to(device).cpu().detach().numpy()
example_output = model(image_batch)
_, y_hat = torch.max(example_output, 1)

for i in range(sample_size):
    plt.title('{}/{}: Label is {}'.format(i+1, sample_size, y_hat[i].cpu().numpy()))
    plt.imshow(x[i].reshape(28, 28), cmap='gray')
    plt.show()
