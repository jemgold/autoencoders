import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConditionalVariationalAutoencoder(nn.Module):
    def __init__(self, n_classes=1, input_size=784,
                 hidden_dim=400, latent_dim=2):
        super(ConditionalVariationalAutoencoder, self).__init__()

        self.input_size = input_size
        self.n_classes = n_classes

        self.fc1 = nn.Linear(input_size + n_classes, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_sigma = nn.Linear(hidden_dim, latent_dim)

        self.fc3 = nn.Linear(latent_dim + n_classes, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_size)

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

    def forward(self, x, category):
        """
        Expects a 2D tensor (N x sz), not a 4D image tensor
        """
        mu, logvar = self.encode(torch.cat((x, category), dim=1))
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(torch.cat((z, category), dim=1))

        return decoded, mu, logvar


def train(batch_size=512, epochs=100):
    from torch.autograd import Variable
    from ignite.trainer import Trainer, TrainingEvents
    import logging
    logger = logging.getLogger('ignite')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    from tensorboardX import SummaryWriter
    from autoencoders.models.sampling import cvae_reconstructions
    from autoencoders.models.utils import flatten, to_one_hot
    from autoencoders.data.mnist import mnist_dataloader
    from autoencoders.utils.tensorboard import run_path
    from autoencoders.models.loss import VAELoss
    use_gpu = torch.cuda.is_available()

    writer = SummaryWriter(run_path('vae'))

    model = ConditionalVariationalAutoencoder(n_classes=10)
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

        inputs, targets = batch
        inputs = flatten(Variable(inputs))
        targets = Variable(to_one_hot(
            targets, batch_size=batch_size, n_classes=10))

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        output, mu, logvar = model(inputs, targets)
        loss = criterion(output, inputs, mu, logvar)
        loss.backward()
        optimizer.step()

        return loss.data[0]

    def validation_inference_function(batch):
        model.eval()

        inputs, targets = batch
        inputs = flatten(Variable(inputs))
        targets = Variable(to_one_hot(
            targets, batch_size=batch_size, n_classes=10))

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        output, mu, logvar = model(inputs, targets)
        loss = criterion(output, inputs, mu, logvar)

        return loss.data[0]

    # def on_end_epoch(state):

    #     meter_loss.reset()

    def on_epoch_competed(trainer, writer):
        print(trainer.current_epoch, np.mean(trainer.training_history))
        writer.add_scalar(
            'loss/training loss', np.mean(trainer.training_history), trainer.current_epoch)

    def on_validation(trainer, writer):
        print(trainer.current_epoch, np.mean(trainer.validation_history))
        writer.add_scalar('loss/validation loss',
                          np.mean(trainer.validation_history), trainer.current_epoch)
        writer.add_image('image', cvae_reconstructions(
            model, val_loader), trainer.current_epoch)

    trainer = Trainer(train_loader, training_update_function,
                      val_loader, validation_inference_function)

    trainer.add_event_handler(TrainingEvents.TRAINING_EPOCH_COMPLETED,
                              on_epoch_competed, writer)
    trainer.add_event_handler(
        TrainingEvents.VALIDATION_COMPLETED, on_validation, writer)

    trainer.run(max_epochs=epochs)


if __name__ == '__main__':
    train(epochs=1, batch_size=64)
