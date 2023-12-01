import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any
from torch.utils.checkpoint import checkpoint_sequential
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import torch.nn.functional as F
from transformers import (
    HfArgumentParser,
    set_seed
)
from collections import Counter
set_seed(100)
from transformers import __version__
from transformers.configuration_utils import PretrainedConfig

from transformers import Trainer
from transformers.deepspeed import deepspeed_init

from transformers.utils import logging
from transformers.trainer_utils import (
    EvalLoopOutput,
    ShardedDDPOption,
    speed_metrics,
    TrainOutput,
    get_last_checkpoint,
    set_seed
)
from transformers.trainer_callback import (
    TrainerState,
)

from transformers.integrations import (
    is_fairscale_available,
    hp_params
)
from transformers.trainer_pt_utils import (
    get_parameter_names,
    IterableDatasetShard
)
from transformers.optimization import (
    Adafactor,
    AdamW
)
from transformers.file_utils import WEIGHTS_NAME
from sklearn.metrics import classification_report
import torch.distributed as dist


logger = logging.get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOFT_MASK_LABELS = "extra_embeddings"


class BaseTrainer(Trainer):
    def __init__(self, eval_targets=None, task=None, metrics=None, extra_info=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_targets = eval_targets
        self.task = task
        self.metrics = metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        # print(inputs.keys())
        targets = inputs.get("target")
        targets = targets.unsqueeze(1)
        outputs = model(**inputs)
        # print(outputs,targets)
        logits = outputs
        # loss = self.mse_loss(logits, targets)
        loss = self.cosent_loss(logits.squeeze(), targets.squeeze())
        return (loss, outputs) if return_outputs else loss
    
    def cosent_loss(self, y_pred, y_true):
        # print(y_pred, y_true)
        y_pred = y_pred[:, None] - y_pred[None, :] 
        y_true = y_true[:, None] - y_true[None, :]
        
        mask = y_true <= 0
        y_true[mask] = -1e12

        loss_matrix = -(y_pred-0.25*y_true)
        loss_matrix = loss_matrix.clamp(min=0)
        del y_pred, y_true
        return torch.mean(loss_matrix)*2

        return torch.log(1+torch.sum(exp_loss))

    def mse_loss(self, logits, targets):
        loss_fct = nn.MSELoss(reduction='none')
        # 分类的loss
        loss = loss_fct(logits.view(-1, 1), targets.view(-1,1) )
        loss = torch.mean(loss)
        return loss

    def evaluate(
        self,
        eval_datasets: Optional[Dataset] = None,
        eval_targets: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = 'eval'
    ) -> Dict[str, float]:
        output = self.eval_loop(
            eval_datasets = eval_datasets,
            eval_targets = eval_targets,
            description="Evaluation",
        )
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        # print(output.metrics)
        return output.metrics
    

    def eval_loop(
        self,
        eval_datasets,
        eval_targets,
        description: str,
        metric_key_prefix: str = "eval",

    ) -> EvalLoopOutput:
        logger.info(f"***** Running {description} *****")
        model = self._wrap_model(self.model, training=False)
        model = model.to(self.args.device)
        eval_datasets = eval_datasets if eval_datasets is not None else self.eval_dataset
        eval_targets = eval_targets if eval_targets is not None else self.eval_targets 
        num_samples = eval_datasets[0].num_rows if isinstance(eval_datasets, list) else eval_datasets.num_rows
        model.eval()
        metrics = self.compute_pet_metrics(eval_datasets, model,)
        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)

    
    def compute_pet_metrics(self, eval_datasets, model):
        dataloader = self.get_eval_dataloader(eval_datasets)
        y_hats = []
        scores = []
        for _, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                labels_ = inputs["label"]
                # print(labels_)
                logits = model(**inputs)
                # print(y_hat.shape, labels_.shape)
                # print(logits, y_hat)
                y_hats.extend(logits.cpu().detach().numpy())
                scores.extend(labels_.cpu().detach().numpy())
                # print(logits)
        rank = self.args.local_rank
        if rank==-1:
            results = {}
            for metric in self.metrics[:1]:
                results.update(metric(scores, y_hats))
            for metric in self.metrics[1:]:
                results.update(metric(scores, y_hats))
            print('*'*15,'Computing metric','*'*15)
            avg_score = sum(results.values())/2
            results['average'] = avg_score
            print(results)
            return results

        device = torch.device(f'cuda:{rank}')

        y_hats_tensor = torch.tensor(np.array(y_hats)).to(device)
        scores_tensor = torch.tensor(np.array(scores)).to(device)

        # 创建存储所有进程数据的Tensor
        y_hats_gather = [torch.zeros_like(y_hats_tensor) for _ in range(dist.get_world_size())]
        scores_gather = [torch.zeros_like(scores_tensor) for _ in range(dist.get_world_size())]
        
        results = {}
        # 收集所有进程的数据
        dist.all_gather(y_hats_gather, y_hats_tensor)
        dist.all_gather(scores_gather, scores_tensor)
        if rank == 0:
            y_hats_gather = torch.cat(y_hats_gather).cpu().tolist()
            scores_gather = torch.cat(scores_gather).cpu().tolist()
            # if the explicit call to wait_stream was omitted, the output below will be
            # non-deterministically 1 or 101, depending on whether the allreduce overwrote
            # the value after the add completed.
            print('*'*15,'Computing metric','*'*15)
            for metric in self.metrics[:1]:
                results.update(metric(scores_gather, y_hats_gather))
            for metric in self.metrics[1:]:
                results.update(metric(scores_gather, y_hats_gather))
            avg_score = sum(results.values())/2
            results['average'] = avg_score
            print(results)
        torch.distributed.barrier()
        results = self.board_cast_dict(results, device)
        return results
    
    def board_cast_dict(self, result, device):
        result_tensor = torch.Tensor([1.0,2.0,3.0]).to(device)
        if dist.get_rank() == 0:
            result_tensor[0] = result['auc']
            result_tensor[1] = result['4_class_acc']
            result_tensor[2] = result['average']
        # 在其他进程中，创建一个空的Tensor来接收数据
        dist.broadcast(result_tensor, src=0)
        result_tensor = result_tensor.cpu().tolist()
        result = {'4_class_acc': result_tensor[1], 'auc': result_tensor[0], 
                  'average': result_tensor[2]}
        return result

    def merge_classes(self, logits):
        merged_logits = np.zeros((logits.shape[0]))  # 创建一个新的数组来保存合并后的概率结果
        merged_logits = logits[:, 2] + logits[:, 3]  # 将类别3和类别4的概率相加作为第二维
        return merged_logits.cpu()




class Predictor(Trainer):
    def __init__(self, eval_targets=None, task=None, metrics=None, extra_info=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_targets = eval_targets
        self.task = task
        self.metrics = metrics
    
    def predict(
        self,
        pred_datasets: Optional[Dataset] = None,
        pred_targets: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = 'pred'
    ) -> Dict[str, float]:
        embs, masks, id_types, scores, labels = self.pred_loop(
            pred_datasets = pred_datasets,
            pred_targets = pred_targets,
            description="Prediction",
        )
        output = self.callback_handler.on_predict(self.args, self.state, self.control,
                                                  (embs, masks, id_types, scores, labels)
                                                  )

        # print(output.metrics)
        return output
    

    def pred_loop(
        self,
        pred_datasets,
        pred_targets,
        description: str,
        metric_key_prefix: str = "pred",

    ) :
        logger.info(f"***** Running {description} *****")
        model = self._wrap_model(self.model, training=False)
        model = model.to(self.args.device)
        # pred_datasets = pred_datasets if pred_datasets is not None else self.pred_dataset
        # pred_targets = pred_targets if pred_targets is not None else self.pred_targets 
        model.eval()
        embs, masks, id_types, scores, labels = self.compute_distill_logits(pred_datasets, model,)
        # Prefix all keys with metric_key_prefix + '_'

        return embs, masks, id_types, scores, labels

    
    def compute_distill_logits(self, eval_datasets, model):
        dataloader = self.get_eval_dataloader(eval_datasets)
        labels = []
        scores = []
        embs = []
        masks = []
        id_types = []
        for _, inputs in enumerate(tqdm(dataloader)):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                labels_ = inputs["target"]
                # print(labels_)
                logits = model(**inputs)
                # print(y_hat.shape, labels_.shape)
                # print(logits, y_hat)
                embs.extend(inputs["input_ids"].cpu().detach().tolist())
                masks.extend(inputs["attention_mask"].cpu().detach().tolist())
                id_types.extend(inputs["token_type_ids"].cpu().detach().tolist())
                scores.extend(logits.cpu().detach().tolist())
                labels.extend(labels_.cpu().detach().tolist())
                # print(logits)
        
        return embs, masks, id_types, scores, labels
    
    def merge_classes(self, logits):
        merged_logits = np.zeros((logits.shape[0]))  # 创建一个新的数组来保存合并后的概率结果
        merged_logits = logits[:, 2] + logits[:, 3]  # 将类别3和类别4的概率相加作为第二维
        return merged_logits.cpu()