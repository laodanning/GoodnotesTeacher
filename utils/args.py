# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from transformers import TrainingArguments
from transformers.deepspeed import HfDeepSpeedConfig
from typing import Optional
import sys
import parser
import os
from transformers import HfArgumentParser

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from base, megatron"}
        # + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
                    "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
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

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

@dataclass
class My_training_args(TrainingArguments):
    is_freeze: bool = field(default=True, metadata={
        "help": "If sets, freeze the transformers parameters"})
    
    predict_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to save prediction results"
        },
    )

    task: Optional[str] = field(
        default="run",
        metadata={"help": "Training task, select from ['train', 'post', 'distill', 'predict']"},
    )

    is_ddp: bool = field(
        default=False,
        metadata={
            "help": "Is using ddp for training."
        },
    )

    loss_ratio: float = field(
        default=0,
        metadata={
            "help": "soft label : hard label's loss weight"
        },
    )
    
@dataclass
class DataArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    train_datasets: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to train dataset"
        },
    )
    
    eval_datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Path to train dataset"}
        # + ", ".join(MODEL_TYPES)},
    )

    test_datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Path to test dataset"}
        # + ", ".join(MODEL_TYPES)},
    )

    datasets_save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save dataset"}
        # + ", ".join(MODEL_TYPES)},
    )

    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether to use cache to save."},
    )
   
    datasets_cache_dir: bool = field(
        default=True,
        metadata={"help": "Directory to save cache."},
    )

    tokenizer_length: Optional[int] = field(
        default=100,
        metadata={"help": "Max length for tokenizer"},
    )
    
    datasets_seed: Optional[int] = field(
        default=100,
        metadata={"help": "Random seed for tokenizer"},
    )

    num_proc: Optional[int] = field(
        default=80,
        metadata={"help": "Number of proc for datasets"},
    )

if __name__=='__main__':
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        parser = HfArgumentParser((My_training_args, ModelArguments, DataArguments))
        training_args,model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        print(data_args.train_datasets)
        print(training_args)