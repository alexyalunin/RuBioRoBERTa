#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from sklearn import metrics as sk_metrics

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.12.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "RuMedDaNet": ("context", "question", "answer"),
    "RuMedTop3": ("symptoms", None, "code"),
    "RuMedSymptomRec": ("symptoms", None, "code"),
    "RuMedNLI": ("ru_sentence1", "ru_sentence2", "gold_label")
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    # label_to_id_file: Optional[str] = field(
    #     default='',
    #     metadata={"help": "top3 default"},
    # )
    multiclass: Optional[bool] = field(
        default=False,
        metadata={"help": ""},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args, data_args)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler('models/log.txt')],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    data_files = {}
    if data_args.train_file:
        data_files['train'] = data_args.train_file
    if data_args.validation_file:
        data_files['validation'] = data_args.validation_file
    if data_args.test_file:
        data_files['test'] = data_args.test_file

    if data_args.task_name is not None:
        data_files['train'] = f'RuMedBench-draft/data/{data_args.task_name}/train_v1.jsonl'
        data_files['validation'] = f'RuMedBench-draft/data/{data_args.task_name}/dev_v1.jsonl'
        data_files['test'] = f'RuMedBench-draft/data/{data_args.task_name}/test_v1.jsonl'

        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    else:
        raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)

    if ('train' in raw_datasets and 'TEXT' in raw_datasets.column_names['train']) or \
            ('validation' in raw_datasets and 'TEXT' in raw_datasets.column_names['validation']):
        raw_datasets = raw_datasets.rename_column("TEXT", "body")
        raw_datasets = raw_datasets.rename_column("LABELS", "label")
    if "validation" not in raw_datasets.keys():
        raw_datasets = raw_datasets.shuffle(seed=training_args.seed)
        raw_datasets["validation"] = load_dataset(
            "csv",
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
        )
        raw_datasets["train"] = load_dataset(
            "csv",
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir
        )

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        # RuMedBench
        sentence1_key, sentence2_key, label_key = task_to_keys[data_args.task_name]
        # raw_datasets = raw_datasets.rename_column(label_key, "label")
        raw_datasets = raw_datasets.map(lambda example: {'label': example[label_key]})
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()
        label_to_id = {v: i for i, v in enumerate(label_list)}
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        if 'train' in raw_datasets:
            non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        else:
            non_label_column_names = [name for name in raw_datasets["validation"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            sentence1_key, sentence2_key = "body", None

        if sentence1_key == 'body' and sentence2_key is None:
            # import joblib
            # label_to_id = joblib.load(data_args.label_to_id_file)
            pass
        elif sentence1_key == "sentence1" and sentence2_key == "sentence2":
            label_to_id = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        else:
            raise

    num_labels = len(label_to_id)
    # num_labels = -1
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.model_name_or_path == 'sberbank-ai/ruBert-base':
        config = transformers.BertConfig.from_pretrained('sberbank-ai/ruBert-base', num_labels=num_labels)
        model = transformers.BertForSequenceClassification.from_pretrained('sberbank-ai/ruBert-base', config=config,
                                                                           cache_dir=model_args.cache_dir)
        tokenizer = transformers.BertTokenizer.from_pretrained('sberbank-ai/ruBert-base', do_lower_case=False)
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            # do_lower_case=do_lower_case,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        import json
        # tok_path = os.path.join(model_args.model_name_or_path, 'tokenizer_config.json')
        # if os.path.isfile(tok_path):
        #     with open(tok_path, 'r', encoding='utf-8') as file:
        #         tok_dict = json.loads(file.read())
        #     do_lower_case = tok_dict["do_lower_case"]

        with open('do_lower_case.json', 'r',
                  encoding='utf-8') as file:
            do_lower_case_dict = json.loads(file.read())
        if model_args.model_name_or_path in do_lower_case_dict:
            do_lower_case = do_lower_case_dict[model_args.model_name_or_path]
            tokenizer.do_lower_case = do_lower_case


    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    print('max_seq_length', max_seq_length)

    def preprocess_function(examples):
        def get_one_hot(labels):
            # a = [0] * num_labels
            # a[label_to_id[label]] = 1
            # return a
            multilabel_vec = [0] * num_labels
            if labels is None:
                labels = ''
            for label in labels.split(';'):
                if label in label_to_id.keys():
                    code = int(label_to_id[label])
                    multilabel_vec[code] = 1
            return multilabel_vec

        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            if data_args.multiclass:
                result["label"] = [get_one_hot(l) for l in examples["label"]]
            else:
                result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
            num_proc=data_args.preprocessing_num_workers,
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    metric_acc = load_metric("accuracy")
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        def hit_at_n(y_true, y_pred, n=3):
            assert len(y_true) == len(y_pred)
            hit_count = 0
            for l, row in zip(y_true, y_pred):
                order = set((np.argsort(row)[::-1])[:n])
                hit_count += int(l in order)
            return hit_count / float(len(y_true))

        pred_probs = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        pred = np.argmax(pred_probs, axis=1)
        true = p.label_ids
        result = metric_acc.compute(predictions=pred, references=true)
        f1_macro = sk_metrics.f1_score(true, pred, average='macro') * 100
        f1_weighted = sk_metrics.f1_score(true, pred, average='weighted') * 100
        result['f1_macro'] = float(f1_macro)
        result['f1_weighted'] = float(f1_weighted)
        if sentence2_key is None:
            for i in [1, 3, 5, 10]:
                score = hit_at_n(true, pred_probs, n=i) * 100
                result['hit@{}'.format(i)] = score
        # print(sk_metrics.classification_report(true, pred, digits=4))
        return result

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # optimizers=(optimizer, scheduler)
    )

    # Training
    if training_args.do_train:
        logger.info(f"*** Training on {data_files['train']} ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        logger.info(f"*** {data_files['validation']} ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            metrics["validation_file"] = data_files['validation']

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    # Evaluation
    if training_args.do_predict:
        logger.info("*** Predict ***")
        logger.info(f"*** {data_files['test']} ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]

        for predict_dataset, task in zip(predict_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=predict_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(predict_dataset)
            )
            metrics["test_samples"] = min(max_eval_samples, len(predict_dataset))
            metrics["test_file"] = data_files['test']

            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

        # write predictions
        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions

            df = predict_dataset.to_pandas()
            df.drop(columns=['attention_mask', 'input_ids'], inplace=True)
            if data_args.task_name in ['RuMedTop3', 'RuMedSymptomRec']:
                top_n = 3
                res = []
                for row in predictions:
                    codes = (np.argsort(row)[::-1])[:top_n]
                    res.append([model.config.id2label[c] for c in codes])
                df['prediction'] = res
            else:
                predictions = np.argmax(predictions, axis=1)
                df['prediction'] = predictions
                df['prediction'] = df['prediction'].apply(lambda x: model.config.id2label[x])
            df.to_json(os.path.join(training_args.output_dir, f"{task}.jsonl"), orient='records', lines=True, force_ascii=False)
            # df.to_csv(os.path.join(training_args.output_dir, f"{task}.csv"), index=False)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
