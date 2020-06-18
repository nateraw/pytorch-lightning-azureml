import glob
import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Union

import nlp
import numpy as np
import pandas as pd
import pyarrow
import torch
from torch.utils.data import DataLoader
from transformers.data import glue_tasks_num_labels

from lightning_base import BaseTransformer, LoggingCallback, set_seed
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Glue tasks in nlp library have names without dashes, so we remove them
glue_tasks_num_labels = {k.replace("-", ""): v for k, v in glue_tasks_num_labels.items()}


class GLUETransformer(BaseTransformer):

    mode = "sequence-classification"

    def __init__(self, hparams: Namespace):
        """A Pytorch Lightning Module for training/evaluating transformers models on the GLUE benchmark

        Args:
            hparams (Namespace): Supplied CLI arguments to configure the run.
        """
        # added to mitigate "can't pickle _thread.lock objects" problem
        delattr(hparams, "logger")
        delattr(hparams, "checkpoint_callback")

        num_labels = glue_tasks_num_labels[hparams.task]
        hparams.glue_output_mode = "classification" if num_labels > 1 else "regression"

        super().__init__(hparams, num_labels, self.mode)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):

        outputs = self(**batch)
        loss = outputs[0]

        tensorboard_logs = {"loss": loss, "rate": self.lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def prepare_data(self):

        if self.hparams.task == "mnli":
            splits = ["train", "validation_matched", "test_matched", "validation_mismatched", "test_mismatched"]
        else:
            splits = ["train", "validation", "test"]
        print("Available Files:\n:", ",\n".join(os.listdir(self.hparams.data_dir)))
        dataset = {}
        for split in splits:
            cached_split_file = self._feature_file(split)
            print(f"LOADING {split} FROM FILE: {cached_split_file}")
            split_dataset = torch.load(cached_split_file)
            dataset[split] = split_dataset
        self.dataset = dataset
        for k, v in self.dataset.items():
            print(f"\t- DATASET SPLIT {k} HAS LENGTH OF {len(v)}")

    def get_dataloader(self, mode: str, batch_size: int) -> Union[List[DataLoader], DataLoader]:
        """Get DataLoader(s) corresponding to the given split 'mode'

        Args:
            mode (str): The dataset split ('train', 'validation', 'test')
            batch_size (int): Number of examples to feed to model on each step.

        Returns:
            Union[List[DataLoader], DataLoader]: Single loader or list of loaders if MNLI
        """

        # Return two dataloaders for val/test datasets if MNLI
        if self.hparams.task == "mnli" and mode in ["validation", "test"]:
            return [
                DataLoader(
                    self.dataset[mode + "_matched"], batch_size=batch_size, num_workers=self.hparams.num_workers,
                ),
                DataLoader(
                    self.dataset[mode + "_mismatched"], batch_size=batch_size, num_workers=self.hparams.num_workers,
                ),
            ]

        # Otherwise, just return a single dataset for the given split mode
        return DataLoader(
            self.dataset[mode],
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=bool(mode == "train"),
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        idx = batch.pop("idx").detach().cpu().numpy()
        return {"idx": idx, **self._eval_step(batch, batch_idx, split="test")}

    def validation_epoch_end(self, outputs: list) -> dict:

        # For MNLI, we have to run self._eval_end twice for matched and mismatched splits
        if self.hparams.task == "mnli":
            logs = {}
            for i, split in enumerate(["matched", "mismatched"]):
                split_logs, _ = self._eval_end(outputs[i])
                # Convert 'validation_loss' to 'validation_loss_matched' or 'validation_loss_mismatched'
                logs.update({k + f"_{split}": v for k, v in split_logs.items()})

            # HACK - pytorch lightning is looking for 'val_loss' as a key in returned dict.
            # Here, we take mean of the matched and mismatched losses and return that as val_loss
            # Is there a better way to do this?
            logs["val_loss"] = torch.mean(torch.stack((logs["val_loss_matched"], logs["val_loss_mismatched"])))

        # For all other datasets, we can simply run self._eval_end once
        else:
            logs, _ = self._eval_end(outputs)

        # Update return dict by adding tensorboard logs and progress bar updates
        logs.update({"log": {**logs}, "progress_bar": {**logs}})
        return logs

    def test_epoch_end(self, outputs):

        # For MNLI, store predictions as Dict[List] where keys are matched and mismatched
        if self.hparams.task == "mnli":
            self.predictions = {}
            self.idxs = {}
            for i, split in enumerate(["matched", "mismatched"]):
                logs, split_preds, split_idxs = self._eval_end(outputs[i], split="test")
                self.predictions[split] = split_preds
                self.idxs[split] = split_idxs

        # Otherwise, store predictions as List
        else:
            logs, self.predictions, self.idxs = self._eval_end(outputs, split="test")

        return logs

    def _eval_step(self, batch, batch_idx, split="val"):
        outputs = self(**batch)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = batch["labels"].detach().cpu().numpy()

        return {f"{split}_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs, split="val"):
        val_loss_mean = torch.stack([x[f"{split}_loss"] for x in outputs]).mean().detach().cpu()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)

        if self.hparams.glue_output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.hparams.glue_output_mode == "regression":
            preds = np.squeeze(preds)

        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        results = {f"{split}_loss": val_loss_mean}

        to_return = (results, preds)

        # For validation dataset, include metric results.
        if split != "test":
            # HACK - to avoid pickle error I didn't assign this as class attribute
            metric = nlp.load_metric("glue", name=self.hparams.task)
            # HACK - the .tolist() call here is to prevent an error:
            # pyarrow.lib.ArrowInvalid: Floating point value truncated error
            results.update(metric.compute(preds.tolist(), out_label_ids.tolist()))

        # Test dataset should include idxs for submission
        else:
            idxs = np.concatenate([x["idx"] for x in outputs], axis=0)
            to_return += (idxs,)

        return to_return

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--task", type=str, required=True, help="The GLUE task to run",
        )
        parser.add_argument(
            "--data_dir", default="./glue_dir", type=str, help="Directory to save/load processed cache data"
        )
        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )
        return parser


def parse_args():
    parser = ArgumentParser()

    # add some script specific args
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default="", help="Directory to write outputs to or load checkpoint from"
    )
    parser.add_argument("--do_train", action="store_true", help="Run training loop")
    parser.add_argument("--do_predict", action="store_true", help="Run test loop")

    # enable all trainer args
    parser = Trainer.add_argparse_args(parser)

    # add the base module args
    parser = BaseTransformer.add_model_specific_args(parser)

    # add the glue module args
    parser = GLUETransformer.add_model_specific_args(parser)

    # cook them all up :)
    args = parser.parse_args()

    return args
