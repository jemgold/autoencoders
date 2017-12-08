import torch
from torch import nn
from autoencoders.models.utils import Lambda, Flatten


class DeepAutoencoder(nn.Module):

    def __init__(self, encoding_dims=[128, 64, 32], input_channels=1,
                 input_size=(28, 28)):
        super(DeepAutoencoder, self).__init__()

        in_channels = input_channels * input_size[0] * input_size[1]

        encoder_blocks = []
        in_size = in_channels
        for output_size in encoding_dims:
            encoder_blocks.append(nn.Linear(in_size, output_size))
            encoder_blocks.append(nn.ReLU(True))
            in_size = output_size

        self.encoder = nn.Sequential(
            Flatten,
            *encoder_blocks
        )

        decoder_blocks = []
        in_size = encoding_dims[-1]
        for output_size in encoding_dims[1::-1]:
            decoder_blocks.append(nn.Linear(in_size, output_size))
            decoder_blocks.append(nn.ReLU(True))
            in_size = output_size

        decoder_blocks.append(nn.Linear(in_size, in_channels))
        decoder_blocks.append(nn.Sigmoid())
        decoder_blocks.append(Lambda(lambda x:
                                     x.view(x.size(0),
                                            input_channels,
                                            input_size[0],
                                            input_size[1])))
        self.decoder = nn.Sequential(*decoder_blocks)

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

    writer = SummaryWriter(run_path('deep'))

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()

    model = DeepAutoencoder()
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
