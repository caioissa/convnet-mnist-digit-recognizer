import sys

import torch
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.optim as optim

from get_data import get_data
from model import Model

def train(model, device, train_loader, optimizer, epoch, number_of_epochs):
    model.train()
    dataset_size = len(train_loader.dataset)
    number_of_batches = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        batch_size = len(data)
        current_loss = loss.item()
        if batch_idx % 100 == 0:
            print('Train Epoch: {}/{} [ {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch+1, number_of_epochs,
                batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/number_of_batches, current_loss))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)))

def check_for_GPU():
    use_cuda = torch.cuda.is_available()
    print('Using GPU: {}'.format(use_cuda))
    return torch.device('cuda' if use_cuda else 'cpu')

def get_epochs():
    args = sys.argv
    if len(args) != 2:
        print('Please include number of epochs as an argument')
        exit()
    else:
        return int(args[1])

def main():
    number_of_epochs = get_epochs()

    dataloaders = get_data()

    device = check_for_GPU()

    model = Model().to(device)
    model.load()

    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    print('Number of epochs: {}'.format(number_of_epochs))
    for epoch in range(number_of_epochs):
        train(model, device, dataloaders['train'], optimizer, epoch, number_of_epochs)
        test(model, device, dataloaders['val'])
        scheduler.step()
        model.save()

if __name__ == '__main__':
    main()
