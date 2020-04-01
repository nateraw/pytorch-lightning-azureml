import argparse
import os
import shutil

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from mnist_model import MNISTModel

from azureml.core.run import Run

# get the Azure ML run object
run = Run.get_context()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_gpus',
        type=int,
        default=1,
        help='number of epochs to train'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='output directory'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.02,
        help='learning rate'
    )

    args = parser.parse_args()

    # Configure early stopping based on average validation loss
    early_stop_callback = EarlyStopping(
        monitor='avg_val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )

    # Init Trainer
    trainer = pl.Trainer(
        early_stop_callback=early_stop_callback,
        gpus=args.n_gpus,
    )
    # Init Model
    model = MNISTModel(args)
    # Fit Model
    trainer.fit(model)

    # Log the fitted model's best validation loss to AzureML
    run.log('best_val_loss', trainer.early_stop_callback.best)

    # Make outputs directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get best checkpoint file name
    ckpt_filename = os.listdir(trainer.weights_save_path)[0]
    # Join it with the full path to get abspath to ckpt file
    ckpt_path = os.path.join(
        trainer.weights_save_path,
        ckpt_filename
    )

    # Copy checkpoint over to outputs directory
    shutil.copy(ckpt_path, args.output_dir)
