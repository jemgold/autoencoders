import torch
import torch.nn as nn
from autoencoders.models.utils import Lambda, Flatten


class SimpleAutoencoder(nn.Module):
    def __init__(self, encoding_dim=32, input_channels=1, input_size=(28, 28)):
        super(SimpleAutoencoder, self).__init__()
        in_channels = input_channels * input_size[0] * input_size[1]

        self.encoder = nn.Sequential(
            Flatten,
            nn.Linear(in_channels, encoding_dim),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, in_channels),
            nn.Sigmoid(),
            Lambda(lambda x: x.view(x.size(0), input_channels,
                                    input_size[0], input_size[1]))
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train(batch_size=512, epochs=50):
    from torch.autograd import Variable
    import torchnet as tnt
    from torchnet.engine import Engine
    from tensorboardX import SummaryWriter
    from autoencoders.models.sampling import sample
    import autoencoders.data.mnist as mnist
    from autoencoders.utils.tensorboard import run_path

    use_gpu = torch.cuda.is_available()

    writer = SummaryWriter(run_path('simple'))

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()

    model = SimpleAutoencoder(encoding_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    criterion = nn.BCELoss()

    dataloader = mnist(batch_size=batch_size)

    if use_gpu:
        model.cuda()
        criterion.cuda()

    def h(sample):
        inputs, _ = sample

        inputs = Variable(inputs)
        if use_gpu:
            inputs = inputs.cuda()

        output = model(inputs)
        loss = criterion(output, inputs)

        return loss, output

    def on_forward(state):
        meter_loss.add(state['loss'].data[0])

    def on_end_epoch(state):
        writer.add_scalar('loss', meter_loss.value()[0], state['epoch'])
        writer.add_image('image', sample(model, dataloader), state['epoch'])

        meter_loss.reset()

    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(h, dataloader, maxepoch=epochs, optimizer=optimizer)


if __name__ == '__main__':
    train()
