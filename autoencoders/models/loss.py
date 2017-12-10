import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


def kl_divergence(mu, sigma):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return torch.sum(torch.exp(sigma) + mu**2 - 1. - sigma)


def vae_loss(output, target, mu, sigma, kl_loss_weight=0.5):
    recon_loss = F.binary_cross_entropy(
        output, target, size_average=False) / output.size(0)

    kl_loss = kl_loss_weight * kl_divergence(mu, sigma)

    return recon_loss + torch.mean(kl_loss)


class VAELoss(_Loss):
    def __init__(self, kl_loss_weight=0.5):
        super(VAELoss, self).__init__(False)
        self.kl_loss_weight = kl_loss_weight

    def forward(self, output, target, mu, logvar):
        return vae_loss(output, target, mu, logvar, self.kl_loss_weight)
