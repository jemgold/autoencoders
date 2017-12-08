from datetime import datetime


def run_path(exp_name):
    """
    Format a tensorboard run name
    """

    'runs/{}_{}'.format(exp_name, datetime.now().strftime('%b%d_%H-%M-%S'))
