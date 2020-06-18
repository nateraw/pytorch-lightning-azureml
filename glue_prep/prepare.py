import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import nlp
import pyarrow
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    for task in ["cola", "mrpc"]:
        # for task in ["cola", "mnli", "mrpc", "sst2", "stsb", "qqp", "qnli", "rte", "wnli"]:
        dataset = nlp.load_dataset("glue", name=task)

        # We don't know names of text field(s) so we find that here. If multiple, we tokenize text pairs.
        text_fields = [field.name for field in dataset["train"].schema if pyarrow.types.is_string(field.type)]

        def convert_to_features(example_batch):

            # Either encode single sentence or sentence pairs
            if len(text_fields) > 1:
                texts_or_text_pairs = list(zip(example_batch[text_fields[0]], example_batch[text_fields[1]]))
            else:
                texts_or_text_pairs = example_batch[text_fields[0]]

            # Tokenize the text/text pairs
            features = tokenizer.batch_encode_plus(
                texts_or_text_pairs, max_length=args.max_seq_length, pad_to_max_length=True
            )

            # Rename 'label' to 'labels' so we can directly unpack batches into model call function
            features["labels"] = example_batch["label"]

            return features

        # By default, we'll drop text fields, label, and idx. If test dataset, don't drop idx
        drop_cols = text_fields + ["label", "idx"]

        # Process each dataset inplace and set the format to torch tensors
        for split in dataset.keys():
            logger.info(f"Preparing {task} - Split: {split}")
            cached_split_file = (
                Path(args.data_dir) / f"cached_{task}_{split}_{args.model_name_or_path}_{str(args.max_seq_length)}"
            )
            cached_split_file.parent.mkdir(exist_ok=True)
            dataset_split = dataset[split].map(
                convert_to_features,
                remove_columns=drop_cols if not split.startswith("test") else drop_cols[:-1],
                batched=True,
                load_from_cache_file=False,
            )
            dataset_split.set_format(type="torch")
            torch.save(dataset_split, cached_split_file)


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--data_dir",
        default=os.environ.get("AZUREML_DATAREFERENCE_prepared_data"),
        type=str,
        help="Directory to save/load processed cache data",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
