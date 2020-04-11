import argparse
from collections import OrderedDict
import logging
import torch
from pathlib import Path
import os

from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_processors as processors
from transformers import glue_output_modes
from transformers import ALL_PRETRAINED_MODEL_ARCHIVE_MAP, AutoTokenizer
from transformers.modeling_auto import MODEL_MAPPING

import azureml.core


mounted_input_path = os.environ['glue_data']
mounted_output_path = os.environ['AZUREML_DATAREFERENCE_prepared_glue_data']
os.makedirs(mounted_output_path, exist_ok=True)

print("---------------------------")
print("INPUT DATA DIR CONTENTS")
print("---------------------------")
print(os.listdir(mounted_input_path))
logger = logging.getLogger(__name__)

ALL_MODELS = tuple(ALL_PRETRAINED_MODEL_ARCHIVE_MAP)
MODEL_CLASSES = tuple(m.model_type for m in MODEL_MAPPING)

TASK2PATH = OrderedDict({
    "cola": "CoLA",
    "mnli": "MNLI",
    "mnli-mm": "MNLI",
    "mrpc": "MRPC",
    "sst-2": "SST-2",
    "sts-b": "STS-B",
    "qqp": "QQP",
    "qnli": "QNLI",
    "rte": "RTE",
    "wnli": "WNLI"
})


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=args.use_fast_tokenizer
    )
    data_dir = Path(mounted_input_path) / TASK2PATH[args.task]
    processor = processors[args.task]()
    labels = processor.get_labels()
    for mode in ["train", "dev"]:
        cached_features_file = Path(mounted_output_path) / "cached_{}_{}_{}_{}".format(
            args.task if not (args.task == 'mnli-mm' and mode == 'train') else 'mnli',
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length)
        )
        if not cached_features_file.exists() and not args.overwrite_cache:
            logger.info("Creating features from dataset file at %s", data_dir)
            examples = (
                processor.get_dev_examples(data_dir)
                if mode == "dev"
                else processor.get_train_examples(data_dir)
            )
            features = convert_examples_to_features(
                examples,
                tokenizer,
                max_length=args.max_seq_length,
                task=args.task,
                label_list=labels,
                output_mode=glue_output_modes[args.task],
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=tokenizer.pad_token_type_id,
            )
            logger.info("Saving features into cached file %s", cached_features_file.as_posix())
            torch.save(features, cached_features_file.as_posix())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default='bert',
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default='bert-base-cased',
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task",
        default='mrpc',
        type=str,
        help="GLUE task to execute in the list: " + ", ".join(processors),
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=129,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--use_fast_tokenizer", action="store_true", help="Whether to use rust-based tokenizers"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
