import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from autoencoders.models.utils import flatten


class VariationalAutoencoder(nn.Module):
    def __init__(self, in_channels=1, input_size=(28, 28), hidden_dim=400,
                 latent_dim=2):
        super(VariationalAutoencoder, self).__init__()

        self.sz = in_channels * input_size[0] * input_size[1]
        self.in_channels = in_channels
        self.input_size = input_size

        self.fc1 = nn.Linear(self.sz, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_sigma = nn.Linear(hidden_dim, latent_dim)

        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, self.sz)

    def reset_parameters(self):
        self.fc2_mu.weight.data.normal_(0, 0.002)
        self.fc2_mu.bias.data.normal_(0, 0.002)
        self.fc2_sigma.weight.data.normal_(0, 0.002)
        self.fc2_sigma.bias.data.normal_(0, 0.002)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_sigma(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(flatten(x))
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)

        return decoded, mu, logvar


def vae_loss(output, target, mu, logvar):
    recon_loss = F.binary_cross_entropy(
        output, target.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)
#     KLD / 512 * 784
    return recon_loss + kl_loss


def train(batch_size=512, epochs=100):
    from torch.autograd import Variable
    import torchnet as tnt
    from torchnet.engine import Engine
    from tensorboardX import SummaryWriter
    from autoencoders.models.sampling import sample_vae
    import autoencoders.data.mnist as mnist
    from autoencoders.utils.tensorboard import run_path

    use_gpu = torch.cuda.is_available()

    writer = SummaryWriter(run_path('vae'))

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()

    model = VariationalAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)

    dataloader = mnist(batch_size=batch_size)

    if use_gpu:
        model.cuda()
        vae_loss.cuda()

    def h(sample):
        inputs, _ = sample

        inputs = Variable(inputs)
        if use_gpu:
            inputs = inputs.cuda()

        output, mu, logvar = model(inputs)
        loss = vae_loss(output, inputs, mu, logvar)

        return loss, output

    def on_forward(state):
        meter_loss.add(state['loss'].data[0])

    def on_start_epoch(state):
        meter_loss.reset()

    def on_end_epoch(state):
        writer.add_scalar('loss', meter_loss.value()[0], state['epoch'])
        writer.add_image('image', sample_vae(
            model, dataloader), state['epoch'])

        meter_loss.reset()

    # engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(h, dataloader, maxepoch=epochs, optimizer=optimizer)


if __name__ == '__main__':
    train()
