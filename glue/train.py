import glob
import os

from azureml.core.run import Run

import pandas as pd
from lightning_base import set_seed
from lightning_glue import GLUETransformer, parse_args
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

# get the Azure ML run object
run = Run.get_context()


class AzureMLLogger(LightningLoggerBase):
    def __init__(self):
        super().__init__()

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for k, v in {**{"step": step}, **metrics}.items():
            run.log(k, v)

    @property
    def experiment(self):
        return ""

    @property
    def name(self):
        return ""

    @property
    def version(self):
        return ""


def main(args):

    # Init Model
    model = GLUETransformer(args)

    aml_logger = AzureMLLogger()

    # Init Trainer
    trainer = Trainer.from_argparse_args(
        args, logger=aml_logger, default_root_dir=args.output_dir if bool(args.output_dir) else os.getcwd()
    )
    # trainer = Trainer.from_argparse_args(args, default_root_dir='./outputs/')

    # Run training and reload best model from checkpoint
    if args.do_train:

        # Fit the model
        trainer.fit(model)

        # Reload best model from current experiment run's checkpoint directory
        experiment_ckpt_dir = model.trainer.weights_save_path
        checkpoints = list(sorted(glob.glob(os.path.join(experiment_ckpt_dir, "epoch=*.ckpt"), recursive=True)))
        trainer.resume_from_checkpoint = checkpoints[-1]

    # Predict on test split and write to submission files
    if args.do_predict:

        # If we didn't train and and output directory is not supplied, raise an exception
        if not args.do_train and not bool(args.output_dir):
            raise RuntimeError("No output_dir is specified for writing results. Try setting --output_dir flag")
        # If we did train, but an output_dir was supplied, use the output_dir
        elif bool(args.output_dir):
            output_dir = args.output_dir
        # If we did train and no output_dir was supplied, use lightning_logs/version_x directory
        elif args.do_train and not bool(args.output_dir):
            output_dir = os.path.dirname(experiment_ckpt_dir)

        # Make the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Run on test data
        trainer.test(model)

        # Have to split MNLI submission into two files for matched and mismatched
        if args.task == "mnli":
            df_matched = pd.DataFrame({"idx": model.idxs["matched"], "prediction": model.predictions["matched"]})
            df_mismatched = pd.DataFrame(
                {"idx": model.idxs["mismatched"], "prediction": model.predictions["mismatched"]}
            )
            df_matched.to_csv(os.path.join(output_dir, "mnli_matched_submission.csv"), index=False)
            df_mismatched.to_csv(os.path.join(output_dir, "mnli_mismatched_submission.csv"), index=False)

        # All other tasks have single submission files
        else:
            df = pd.DataFrame({"idx": model.idxs, "prediction": model.predictions})
            df.to_csv(os.path.join(output_dir, f"{args.task}_submission.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    args.do_train = True
    args.do_predict = True
    args.output_dir = "./outputs"
    args.data_dir = os.environ["AZUREML_DATAREFERENCE_prepared_data"]
    run.log("task", args.task)
    run.log("model_name_or_path", args.model_name_or_path)
    run.log("learning_rate", args.learning_rate)
    run.log("seed", args.seed)
    set_seed(args)
    main(args)
