import torch
from torchvision import datasets, transforms


def mnist_dataloader(path='../data', batch_size=128, train=True, download=False):
    dataset = datasets.MNIST(
        path, transform=transforms.ToTensor(), download=download, train=train)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size)

    return dataloader
