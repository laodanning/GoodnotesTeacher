# -*- coding: utf-8 -*-
# 奇怪的bug，先import pandas后import torch会报错，不要改变这边import pkg的顺序，不要改变！
import torch

if torch.cuda.is_available():
    torch.cuda.set_device(0)
import pandas as pd
import os

os.environ['TRANSFORMERS_OFFLINE'] = 'yes'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/search01/usr/laodanning/cache'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed
)

set_seed(100)
torch.manual_seed(3407)
from torch import nn as nn
from transformers import BertConfig, MegatronBertConfig
from transformers import BertPreTrainedModel
from transformers import BertTokenizer, BertModel, PreTrainedTokenizerFast, DataCollatorWithPadding
from model.models import model_dict, RegressionModel
from model.trainer import BaseTrainer
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import lr_scheduler
from model.data_loader import Input_Datasets, Distill_Datasets
from tqdm import tqdm
from utils.args import My_training_args, ModelArguments, DataArguments
from model.metric import metrics_1, metric_auc
from transformers import TrainerCallback
import sys
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from utils.utils import remove_duplicate_in_train
import transformers
# from utils.wobert_tokenizer import WoBertTokenizer
from modeltools import WoBertTokenizer
# import wandb
import time
import warnings

warnings.filterwarnings("ignore")
import logging

logging.getLogger().setLevel(logging.ERROR)

tokenizer_dict = {
    'base': BertTokenizer.from_pretrained("ckiplab/bert-tiny-chinese"),
    'Megatron': WoBertTokenizer.from_pretrained('/mnt/search01/usr/laodanning/relevancebert/checkpoint/bert-48/',
                                                use_fast=True,
                                                )
}

config_dict = {
    'base': BertConfig,
    'Megatron': MegatronBertConfig
}


class SaveMetricResultsCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metric_results = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # wandb.log(metrics)
        self.metric_results.append(metrics)
        # Save the metric results to a file
        self.save_results(args.local_rank)

    def save_results(self, rank):
        # print('saving.......'+'*'*20)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        file_path = os.path.join(self.output_dir, "metric_results.txt")
        with open(file_path, "w") as file:
            for metrics in self.metric_results:
                file.write(str(metrics) + "\n")


def run():
    # 对输入参数进行解析。
    parser = HfArgumentParser((My_training_args, ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        training_args, model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        raise ValueError('Need to provide json configs.')
    rank = training_args.local_rank

    # os.environ["WANDB_PROJECT"] = training_args.task+'-'+time.strftime('%Y-%m-%d-%H-%M-%S ',time.localtime(time.time())) # name your W&B project

    # if rank<=0:
    #     wandb.login()
    #     run = wandb.init(
    #     # Set the project where this run will be logged
    #     project=,
    #     # Track hyperparameters and run metadata
    #     config={
    #         "learning_rate": training_args.learning_rate,
    #         "epochs": training_args.num_train_epochs,
    #         "batch_size":training_args.per_device_train_batch_size*training_args.gradient_accumulation_steps*8,
    #         "output_dir":training_args.output_dir,
    #         "eval_steps":training_args.eval_steps
    #     })

    # 定义tokenizer
    transformers.logging.set_verbosity_error()
    model_name_or_path = model_args.model_name_or_path
    tokenizer = tokenizer_dict[model_args.model_type]
    # 解析输入的数据集
    if rank == -1:
        print(data_args)
        train_dataset, eval_dataset = Distill_Datasets(
            train_dir=data_args.train_datasets,
            test_dir=data_args.eval_datasets,
            data_seed=100,
            tokenizer=tokenizer,
            cache_dir=data_args.datasets_cache_dir,
        ).get_datasets()

    else:
        if rank == 0:
            train_dataset, eval_dataset = Distill_Datasets(
                train_dir=data_args.train_datasets,
                test_dir=data_args.eval_datasets,
                data_seed=100,
                tokenizer=tokenizer,
                cache_dir=data_args.datasets_cache_dir,
            ).get_datasets()
        # 由于datasets会存储cache，所以多进程加载时需要
        torch.distributed.barrier()
        if rank != 0:
            train_dataset, eval_dataset = Distill_Datasets(
                train_dir=data_args.train_datasets,
                test_dir=data_args.eval_datasets,
                data_seed=100,
                tokenizer=tokenizer,
                cache_dir=data_args.datasets_cache_dir,
            ).get_datasets()

    """有时候数据集换来换去，训练集和验证集会有一些重复，在这里加一道检查工序。
    Posttrain与predict时没有eval_datasets"""
    # if training_args.task not in ['post', 'predict']:
    #     train_dataset, eval_dataset = remove_duplicate_in_train(train_dataset, eval_dataset)
        # train_dataset = train_dataset.shuffle(seed=42)

    # 检查一下分词是否正确
    if rank <= 0:
        print('Examples: ', train_dataset[15:16])

    if training_args.local_rank<=0:
        print(train_dataset, eval_dataset)
    train_dataset = train_dataset.shuffle(seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if training_args.local_rank >= 0:
        device = torch.device(f'cuda:{training_args.local_rank}')

    training_args.fp16 = True

    # 加载模型
    bert_config = config_dict[model_args.model_type].from_pretrained(model_name_or_path)
    model = model_dict[model_args.model_type](bert_config)
    model = model.from_pretrained(model_name_or_path,
                                  config=bert_config,
                                  _fast_init=False).to(device)

    output_dir = training_args.output_dir

    # 如果没有进行ddp，则计算lr schedule步数，并自定义optimizer
    if training_args.local_rank == -1:
        training_args.sharded_ddp = ''
        total_samples = training_args.num_train_epochs * train_dataset.num_rows
        sample_per_batch = training_args.per_device_train_batch_size // abs(training_args._n_gpu)
        total_training_steps = total_samples // sample_per_batch + 1
        total_training_steps = total_training_steps // training_args.gradient_accumulation_steps
        print("lr scheduler's total step is %d" % total_training_steps)
        optimizer = AdamW(model.parameters(), lr=training_args.learning_rate,
                          no_deprecation_warning=True)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=total_training_steps * 0.1,
                                                    num_training_steps=total_training_steps)
        trainer = BaseTrainer(
            model=model,
            args=training_args,
            metrics=[metrics_1, metric_auc],
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_targets=eval_dataset,
            tokenizer=tokenizer,
            callbacks=[SaveMetricResultsCallback(output_dir)],
            optimizers=(optimizer, scheduler),
        )

    # 如果进行ddp， 则使用默认的optimizer
    else:
        trainer = BaseTrainer(
            model=model,
            args=training_args,
            metrics=[metrics_1, metric_auc],
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_targets=eval_dataset,
            tokenizer=tokenizer,
            callbacks=[SaveMetricResultsCallback(output_dir)],
        )

    trainer.train()


def _mp_fn(index):
    run()


if __name__ == "__main__":
    run()

