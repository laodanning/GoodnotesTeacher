# 奇怪的bug，先import pandas后import torch会报错，不要改变这边import pkg的顺序，不要改变！
import torch
if torch.cuda.is_available():
    torch.cuda.set_device(0)
import pandas as pd
import os
os.environ['TRANSFORMERS_CACHE'] = '../.cache'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TRANSFORMERS_OFFLINE']='yes'

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed
)
set_seed(100)
torch.manual_seed(3407) 
from torch import nn as nn
from transformers import BertConfig
from transformers import BertPreTrainedModel
from transformers import BertTokenizer, BertModel,PreTrainedTokenizerFast,DataCollatorWithPadding
from model.models import model_dict, RegressionModel
from model.trainer import BaseTrainer
from transformers import AdamW, get_cosine_schedule_with_warmup
from torch.optim import lr_scheduler
from model.data_loader import Input_Datasets, Distill_Datasets
from tqdm import tqdm
from utils.args import My_training_args, ModelArguments, DataArguments
from model.metric import metrics_1, metric_auc, metrics_classification
from transformers import TrainerCallback
import sys
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from utils.utils import remove_duplicate_in_train
import transformers
from transformers import AutoTokenizer
# import datasets
# datasets.set_caching_enabled(False)
import warnings
warnings.filterwarnings("ignore")
import logging

logging.getLogger().setLevel(logging.ERROR)



class SaveMetricResultsCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metric_results = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.metric_results.append(metrics)
        # Save the metric results to a file
        self.save_results()

    def save_results(self):
        # print('saving.......'+'*'*20)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        file_path = os.path.join(self.output_dir, "metric_results.txt")
        with open(file_path, "w") as file:
            for metrics in self.metric_results:
                file.write(str(metrics) + "\n")

def run():
    model_type = 'base_model'
    parser = HfArgumentParser((My_training_args, ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
        training_args, model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        json_path = input("请输入训练参数文件路径，输入空则为configs/train.json:")
        if not json_path:json_path = 'configs/train_small.json'
        training_args, model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(json_path))
    
    model_name_or_path = model_args.model_name_or_path
    # tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained("/home/chen/.fast_cache/models--ckiplab--bert-tiny-chinese/snapshots/ca5496ebfd1b6f7c95740b0a06ecbc43f3135a3b")
    except:
        tokenizer = AutoTokenizer.from_pretrained("../.fast_cache/models--ckiplab--bert-tiny-chinese/snapshots/ca5496ebfd1b6f7c95740b0a06ecbc43f3135a3b")

    # tokenizer.add_special_tokens({'additional_special_tokens':["[/BR]","[/CA]",'[/SK]','[unused1]']})
    
    old_level = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    train_dataset, eval_dataset = Distill_Datasets(
                                    train_dir = data_args.train_datasets,
                                    test_dir = data_args.eval_datasets,
                                    data_seed=100, 
                                    tokenizer = tokenizer,
                                    cache_dir=data_args.datasets_cache_dir,
                                    ).get_datasets()
    if training_args.local_rank<=0:
        print(train_dataset, eval_dataset)
    train_dataset = train_dataset.shuffle(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # select which model architecture to train with, detailed model list in model.models
    bert_config = BertConfig.from_pretrained(model_name_or_path)
    bert_config._name_or_path = model_name_or_path
    # print(bert_config)
    model = model_dict[model_type](bert_config)

    # print(bert_config)
    model = model.from_pretrained(model_name_or_path,
                                  config=bert_config,
                                  _fast_init=False).to(device)
    # model.resize_token_embeddings(len(tokenizer))
    # print(ddp_model)
    
    
    output_dir = "/data/chen/checkpoint/%s"%model_name_or_path
    training_args.dataloader_num_workers = 1
    total_samples = training_args.num_train_epochs*train_dataset.num_rows
    sample_per_batch = training_args.per_device_train_batch_size*training_args._n_gpu
    total_training_steps = total_samples//sample_per_batch+1
    total_training_steps = total_training_steps//training_args.gradient_accumulation_steps
    if training_args.local_rank<=0:
        print(training_args.eval_steps,'-----------------')
        print(training_args.eval_steps,'-----------------')
        print("lr scheduler's total step is %d"%total_training_steps)
        print("lr scheduler's total step is %d"%total_training_steps)
    training_args.fp16 = True
    # model.resize_token_embeddings(len(tokenizer))
    optimizer = AdamW(model.parameters(), lr = training_args.learning_rate, 
                      no_deprecation_warning=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=total_training_steps*0.1,
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
        # optimizers = (optimizer, scheduler),
    )

    trainer.train()
    
    
if __name__ == "__main__":
    # 约20分钟
    run()
    # run(model_name_or_path='hfl/chinese-lert-small')
    # freeze bert参数大约需要40
    # run('hfl/chinese-lert-large')
    # run('hfl/chinese-macbert-base', is_freeze=True)
