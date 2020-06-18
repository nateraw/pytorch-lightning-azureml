import os
from argparse import ArgumentParser

from model import MNISTModel
from pytorch_lightning import Trainer

parser = ArgumentParser()
parser.add_argument("--data_dir", type=str, default=os.environ.get("MNIST_DIR", "."))

parser = Trainer.add_argparse_args(parser)
parser = MNISTModel.add_model_specific_args(parser)
hparams = parser.parse_args()

# Init Model
model = MNISTModel(hparams)

# Init Trainer
trainer = Trainer.from_argparse_args(hparams)

# Fit the Model
trainer.fit(model)
