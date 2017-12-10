from autoencoders.models.ConditionalVariationalAutoencoder import train
from itertools import product

opts = {
    'epochs': [200],
    'batch_size': [64, 256, 512],
    'latent_dim': [2, 24, 48, 128],
    'hidden_dim': [64, 128, 256, 512],
}


def grid_search(d):
    return [dict(zip(d, v)) for v in product(*d.values())]


for experiment, opts in enumerate(grid_search(opts)):
    train(epochs=opts['epochs'], batch_size=opts['batch_size'],
          latent_dim=opts['latent_dim'], hidden_dim=opts['hidden_dim'],
          use_gpu=True)
