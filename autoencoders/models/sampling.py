import torch
from torch.autograd import Variable
import torchvision.utils as vutils

use_gpu = torch.cuda.is_available()


def sample(model, loader, n=10):
    imgs = next(iter(loader))[0][:n]
    imgs = Variable(imgs)
    if use_gpu:
        imgs = imgs.cuda()

    reconstructions = model(imgs)
    reconstructions = reconstructions.view(reconstructions.size(0), 1, 28, 28)

    return vutils.make_grid(torch.cat([imgs.data, reconstructions.data]), n)


def vae_reconstructions(model, loader, n=10):
    model.eval()
    imgs = next(iter(loader))[0][:n]
    imgs = Variable(imgs)
    if use_gpu:
        imgs = imgs.cuda()

    reconstructions, _, _ = model(imgs)
    reconstructions = reconstructions.view(reconstructions.size(0), 1, 28, 28)

    return vutils.make_grid(torch.cat([imgs.data, reconstructions.data]), n)
