import torch
from torch import nn as nn
import os
os.environ['TRANSFORMERS_CACHE'] = '../.cache'
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import MegatronBertPreTrainedModel, MegatronBertModel
BertLayerNorm = torch.nn.LayerNorm

class RegressionModel(BertPreTrainedModel):

    def __init__(self, config: BertConfig, dropout_rate=0.1):
        super(RegressionModel, self).__init__(config)
        # self.bert = BertModel.from_pretrained('hfl/chinese-lert-small', config=config)
        self.bert = BertModel(config)
        self.drop = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.linear_relu_stack = torch.nn.Sequential(self.fc1
                       ,nn.ReLU()
                       ,self.fc2
                       ,nn.ReLU()
                       ,self.fc3,
                       )
        # self.init_weights()

    def forward(self, input_ids=None, 
                attention_mask=None, 
                token_type_ids = None, 
                target=None,teacher=None, *args,
                **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        last_hidden_state = outputs.last_hidden_state  # 最后一层的隐藏状态

        mask = attention_mask.view(-1, attention_mask.shape[1], 1)
        masked_hidden_state = last_hidden_state * mask  # 将隐藏状态与mask相乘，使得padding部分的值为0
        sum_hidden_state = torch.sum(masked_hidden_state, dim=1)  # 按序列长度求和
        sum_mask = torch.sum(mask, dim=1)
        avg_hidden_state = sum_hidden_state / sum_mask  # 按序列长度求平均

        output = self.drop(avg_hidden_state)
        output = self.linear_relu_stack(output)  
        return output

    def resize_token_embeddings(self, resize_length):
        self.bert.resize_token_embeddings(resize_length)
        return

    def init_linear_layers(self):
        print("Reinitializing the linear layers")
        for module in self.linear_relu_stack:
            if isinstance(module, nn.Linear):
                module.reset_parameters()

class MegatronRegressionModel(MegatronBertPreTrainedModel):
    def __init__(self, config: BertConfig, dropout_rate=0.1):
        super(MegatronRegressionModel, self).__init__(config)
        # self.bert = BertModel.from_pretrained('hfl/chinese-lert-small', config=config)
        self.bert = MegatronBertModel(config)
        self.drop = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.linear_relu_stack = torch.nn.Sequential(self.fc1
                       ,nn.ReLU()
                       ,self.fc2
                       ,nn.ReLU()
                       ,self.fc3,
                       )
        # self.init_weights()

    def forward(self, input_ids=None, 
                attention_mask=None, 
                token_type_ids = None, 
                target=None,teacher=None, 
                *args,
                **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        last_hidden_state = outputs.last_hidden_state  # 最后一层的隐藏状态

        mask = attention_mask.view(-1, attention_mask.shape[1], 1)
        masked_hidden_state = last_hidden_state * mask  # 将隐藏状态与mask相乘，使得padding部分的值为0
        sum_hidden_state = torch.sum(masked_hidden_state, dim=1)  # 按序列长度求和
        sum_mask = torch.sum(mask, dim=1)
        avg_hidden_state = sum_hidden_state / sum_mask  # 按序列长度求平均
        
        output = self.drop(avg_hidden_state)
        output = self.linear_relu_stack(output)  
        return output

    def resize_token_embeddings(self, resize_length):
        self.bert.resize_token_embeddings(resize_length)
        return

    def init_linear_layers(self):
        print("Reinitializing the linear layers")
        for module in self.linear_relu_stack:
            if isinstance(module, nn.Linear):
                module.reset_parameters()


model_dict = {
    'base': RegressionModel, 
    'Megatron': MegatronRegressionModel
}

if __name__=='__main__':
    bert_config = BertConfig.from_pretrained('hfl/chinese-lert-small')
    print(bert_config)
    model = RegressionModel(bert_config)
    print(model.parameters)