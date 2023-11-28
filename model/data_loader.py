import pandas as pd
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset
from transformers import BertTokenizer, BertModel
import os

"""使用huggingface Dataset实现划分训练/测试集， 并且使用map函数进行批处理。
更方便添加新功能，更快的思必得！"""


class Input_Datasets(Dataset):
    def __init__(self, train_dir=None, test_dir=None, data_seed=100,
                 tokenizer=BertTokenizer.from_pretrained('hfl/chinese-lert-small')
                 , cache_dir='../datasets_cache', num_proc=80, use_cache=True):
        self.data_seed = data_seed
        self.data_dir = train_dir
        self.cache_dir = cache_dir
        self.tokenizer_length = 450

        if not use_cache:
            cache_dir = None

        self.tokenizer = tokenizer
        # self.tokenizer.add_special_tokens({'additional_special_tokens':["[/BR]","[/CA]",'[/SK]','[unused1]']})
        self.test_dir = test_dir
        self.num_proc = 80

    # 读取数据集
    def load_datasets(self):
        dataset_args = {}
        print("Load train datasets from: %s" % self.data_dir)
        print("Load test datasets from: %s" % self.test_dir)
        train_files = {
            "train": self.data_dir,
        }
        test_files = {
            "test": self.test_dir
        }
        return load_dataset("csv", data_files=train_files, cache_dir=self.cache_dir, **dataset_args, ), load_dataset(
            "csv", data_files=test_files, cache_dir=self.cache_dir, **dataset_args, delimiter='\t', )

    # 对数据进行tokenize，由于datasets可以进行multiprocess，故将此任务放到这。
    def tokenize_example(self, example):
        # 去除query30词以后的内容，超长query会在padding以后拖慢整个batch的计算速度
        target = example['label']
        encoding = self.tokenizer(example['sentence1'][:30],
                                  example['sentence2'],
                                  truncation=True, padding=True,
                                  max_length=self.tokenizer_length,
                                  return_token_type_ids=True)

        return {'input_ids': encoding['input_ids'],
                'token_type_ids': encoding['token_type_ids'],
                'attention_mask': encoding['attention_mask'],
                'target': target}

    # 获得我们需要的datasets
    def get_datasets(self):
        train_datasets, test_datasets = self.load_datasets()
        train_datasets, test_datasets = train_datasets['train'], test_datasets['test']
        train_datasets = train_datasets.map(self.tokenize_example, num_proc=80, )
        test_datasets = test_datasets.map(self.tokenize_example, num_proc=80)
        return train_datasets, test_datasets


class Distill_Datasets(Input_Datasets):
    def tokenize_example_train(self, example):
        # 如果发现为空，则sentence1 2返回一样的内容，这样子最终的得分肯定是1。
        if not isinstance(example['sentence1'], str) or not isinstance(example['sentence2'], str) or not isinstance(
                example['label'], float):
            print(example)
            example['sentence1'] = '无内容'
            example['sentence2'] = '无内容'
            example['label'] = 1

        def strQ2B(ustring):
            """全角转半角"""
            rstring = ""
            for uchar in ustring:
                inside_code = ord(uchar)
                if inside_code == 12288:  # 全角空格直接转换
                    inside_code = 32
                elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
                    inside_code -= 65248

                rstring += chr(inside_code)
            return rstring

        example['sentence1'] = strQ2B(example['sentence1'])
        example['sentence2'] = strQ2B(example['sentence2'])
        encoding = self.tokenizer(example['sentence1'],
                                  example['sentence2'],
                                  truncation=True, padding=True,
                                  max_length=self.tokenizer_length,
                                  return_token_type_ids=True)
        target = example['label']
        return {'input_ids': encoding['input_ids'],
                'token_type_ids': encoding['token_type_ids'],
                'attention_mask': encoding['attention_mask'],
                'target': target}

    # 获得一个目录下面所有csv文件
    def fetch_file_dirs(self, directory_path):
        csv_files = []
        for filename in os.listdir(directory_path):
            if filename.endswith('.csv'):
                csv_files.append(os.path.join(directory_path, filename))

        return csv_files

    def get_data_chunks(self, directory_path):
        csv_files = self.fetch_file_dirs(directory_path)
        # print(csv_files[:2])
        dataset_chunks = []
        for csv_file in csv_files[:]:
            print('Loading %s...', csv_file)
            chunk = load_dataset("csv", data_files=csv_file, cache_dir=self.cache_dir, delimiter='\t')
            chunk = chunk.map(self.tokenize_example_train, num_proc=80, )
            dataset_chunks.append(chunk['train'])

        return dataset_chunks

    # 获得我们需要的datasets
    def get_datasets(self):
        datasets_chunk = self.get_data_chunks(self.data_dir)
        test_datasets = load_dataset("csv", data_files=self.test_dir, cache_dir=self.cache_dir, delimiter='\t', )
        test_datasets = test_datasets['train'].map(self.tokenize_example, num_proc=80)
        train_datasets = concatenate_datasets(datasets_chunk)
        return train_datasets, test_datasets


if __name__ == '__main__':
    a = Input_Datasets(data_dir='./data/goods_detail.csv',
                       test_dir='./data/test_set.csv', data_seed=100)
    train_data, test_data = a.get_datasets()
