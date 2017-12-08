import torch
from torch import nn
from autoencoders.models.utils import xavier_weights_init


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, in_channels=1):
        super(ConvolutionalAutoencoder, self).__init__()

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

        [xavier_weights_init(m) for m in self.modules()]

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train(batch_size=512, epochs=100):
    from torch.autograd import Variable
    import torchnet as tnt
    from torchnet.engine import Engine
    from tensorboardX import SummaryWriter
    from autoencoders.models.sampling import sample
    import autoencoders.data.mnist as mnist
    from autoencoders.utils.tensorboard import run_path

    use_gpu = torch.cuda.is_available()

    writer = SummaryWriter(run_path('conv'))

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()

    model = ConvolutionalAutoencoder()
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

    def on_start_epoch(state):
        meter_loss.reset()

    def on_end_epoch(state):
        writer.add_scalar('loss', meter_loss.value()[0], state['epoch'])
        writer.add_image('image', sample(model, dataloader), state['epoch'])

        meter_loss.reset()

    # engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(h, dataloader, maxepoch=epochs, optimizer=optimizer)


if __name__ == '__main__':
    train()
