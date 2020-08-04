from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

dataset_dir = './data/'
def get_data():
   data_transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,),(0.3081,))
   ])

   dataset = {
       'train': MNIST(dataset_dir, train=True, transform=data_transform, download=True),
       'val': MNIST(dataset_dir, train=False, transform=data_transform, download=True)
   }

   dataloaders = {
       'train': DataLoader(dataset['train'], batch_size=100, shuffle=True, num_workers=4),
       'val': DataLoader(dataset['val'], batch_size=100, shuffle=False, num_workers=4)
   }

   return dataloaders
