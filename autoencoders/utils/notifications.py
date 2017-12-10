import os
from dotenv import load_dotenv, find_dotenv
from simplepush import send

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# load up the entries as environment variables
load_dotenv(dotenv_path)

simplepush_key = os.environ.get('SIMPLEPUSH_KEY')


def send_training_complete_push(title, desc):
    if simplepush_key:
        send(simplepush_key, title, desc, 'trainingComplete')
