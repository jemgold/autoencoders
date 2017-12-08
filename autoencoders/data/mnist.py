import torch
from torchvision import datasets, transforms


def mnist_dataloader(batch_size=128):
    dataset = datasets.MNIST('../data', transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    return dataloader
