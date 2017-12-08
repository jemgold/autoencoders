import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from autoencoders.models.utils import flatten
from autoencoders.utils import xavier_weights_init


class ConvolutionalVariationalAutoencoder(nn.Module):
    def __init__(self, in_channels=1, input_size=(28, 28), hidden_dim=400,
                 latent_dim=2):
        super(ConvolutionalVariationalAutoencoder, self).__init__()

        self.sz = in_channels * input_size[0] * input_size[1]
        self.in_channels = in_channels
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, (3, 3), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 8, (3, 3), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 8, (3, 3), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(8, 8, (3, 3), padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 8, (3, 3), padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 16, (3, 3), padding=0),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, in_channels, kernel_size=(3, 3), padding=1),
            nn.Sigmoid(),
        )

        self.fc21 = nn.Linear(128, latent_dim)
        self.fc22 = nn.Linear(128, latent_dim)

        self.fc3 = nn.Linear(latent_dim, 128)

        [xavier_weights_init(m) for m in self.modules()]

    def encode(self, x):
        h1 = flatten(self.encoder(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h2 = self.fc3(z).view(z.size(0), 8, 4, 4)
        return self.decoder(h2)
#         h3 = F.relu(self.fc3(z))
#         return F.sigmoid(self.fc4(h3))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
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

    writer = SummaryWriter(run_path('conv_vae'))

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()

    model = ConvolutionalVariationalAutoencoder()
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
