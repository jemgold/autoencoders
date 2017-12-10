import torch
from torch import nn
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


def train(batch_size=512, epochs=100):
    from torch.autograd import Variable
    from ignite.trainer import Trainer, TrainingEvents
    import logging
    logger = logging.getLogger('ignite')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    from tensorboardX import SummaryWriter
    from autoencoders.models.sampling import vae_reconstructions
    from autoencoders.data.mnist import mnist_dataloader
    from autoencoders.utils.tensorboard import run_path
    from autoencoders.models.loss import VAELoss
    import numpy as np
    use_gpu = torch.cuda.is_available()

    writer = SummaryWriter(run_path('vae'))

    model = VariationalAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    criterion = VAELoss()

    train_loader = mnist_dataloader(
        path='data', batch_size=batch_size, download=True)

    val_loader = mnist_dataloader(
        path='data', batch_size=batch_size, train=False, download=True)

    if use_gpu:
        model.cuda()
        criterion.cuda()

    def training_update_function(batch):
        model.train()
        optimizer.zero_grad()

        inputs, _ = batch
        inputs = Variable(inputs)
        if use_gpu:
            inputs = inputs.cuda()

        output, mu, logvar = model(inputs)
        loss = criterion(output, inputs.view(-1, 784), mu, logvar)
        loss.backward()
        optimizer.step()

        return loss.data[0]

    def validation_inference_function(batch):
        model.eval()

        inputs, _ = batch
        inputs = Variable(inputs)
        if use_gpu:
            inputs = inputs.cuda()

        output, mu, logvar = model(inputs)
        loss = criterion(output, inputs, mu, logvar)

        return loss.data[0]

    def on_epoch_competed(trainer, writer):
        writer.add_scalar(
            'loss/training loss',
            np.mean(trainer.training_history),
            trainer.current_epoch)

    def on_validation(trainer, writer):
        writer.add_scalar('loss/validation loss',
                          np.mean(trainer.validation_history),
                          trainer.current_epoch)

        writer.add_image('image', vae_reconstructions(
            model, val_loader), trainer.current_epoch)

    trainer = Trainer(train_loader, training_update_function,
                      val_loader, validation_inference_function)

    trainer.add_event_handler(TrainingEvents.TRAINING_EPOCH_COMPLETED,
                              on_epoch_competed, writer)
    trainer.add_event_handler(
        TrainingEvents.VALIDATION_COMPLETED, on_validation, writer)

    trainer.run(max_epochs=epochs)


if __name__ == '__main__':
    train()
