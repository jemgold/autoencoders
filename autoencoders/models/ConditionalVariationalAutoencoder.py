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


def train(epochs=10, batch_size=64, latent_dim=2,
          hidden_dim=400, use_gpu=False):
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

    experiment_name = run_path(
        'cvae_l{}_h{}_b{}_adam_3e-4'.format(latent_dim, hidden_dim, batch_size))

    writer = SummaryWriter(experiment_name)

    model = ConditionalVariationalAutoencoder(
        latent_dim=latent_dim, hidden_dim=hidden_dim, n_classes=10)
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

    def on_complete(trainer):
        checkpoint_path = 'models/{}.cpt'.format(
            experiment_name.split('/')[-1])

        torch.save(model.state_dict(), checkpoint_path)

    trainer = Trainer(train_loader, training_update_function,
                      val_loader, validation_inference_function)

    trainer.add_event_handler(TrainingEvents.TRAINING_EPOCH_COMPLETED,
                              on_epoch_competed, writer)
    trainer.add_event_handler(
        TrainingEvents.VALIDATION_COMPLETED, on_validation, writer)

    trainer.add_event_handler(TrainingEvents.TRAINING_COMPLETED, on_complete)

    trainer.run(max_epochs=epochs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Conditional Variational Autoencoders')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--use_gpu', action="store_true", default=False)

    opts = parser.parse_args()

    train(epochs=opts.epochs, batch_size=opts.batch_size,
          latent_dim=opts.latent_dim, hidden_dim=opts.hidden_dim, use_gpu=opts.use_gpu)
