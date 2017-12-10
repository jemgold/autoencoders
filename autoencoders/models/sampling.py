import torch
from torch.autograd import Variable
import torchvision.utils as vutils
from autoencoders.models.utils import flatten, to_one_hot

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


def cvae_reconstructions(model, loader, n=10):
    model.eval()

    inputs, targets = next(iter(loader))

    batch_size = inputs.size(0)

    n = min(n, batch_size)

    inputs = Variable(inputs)[:n]

    # TODO: remove hardcoded n_classes
    targets = Variable(to_one_hot(
        targets[:n], batch_size=n, n_classes=10))

    if use_gpu:
        inputs = inputs.cuda()
        targets = targets.cuda()

    reconstructions, _, _ = model(flatten(inputs), targets)
    reconstructions = reconstructions.view(reconstructions.size(0), 1, 28, 28)

    return vutils.make_grid(torch.cat([inputs.data, reconstructions.data]), n)
