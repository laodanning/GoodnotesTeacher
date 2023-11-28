import numpy as np
from sklearn.metrics import classification_report
# from data_loader import *
import pandas as pd
from sklearn.metrics import roc_auc_score

"""进行分类，并返回结果
接受的输入为scores:将标注的label映射成的0.15等分数
logits:预测的结果
该函数用于回归函数"""
def metrics_1(scores, logits,  mode='1'):
    scores = np.array(scores)
    logits = np.array(logits)
    label_dict = {0.05:0 ,0.15:1, 0.35:2, 0.65:3, 0.9:4}
    thresholds = [0.1,0.25, 0.5, 0.75]
    logits_class = np.digitize(logits, thresholds)
    def map_value(value):
        for key, label in label_dict.items():
            if np.isclose(value, key):
                return label
        return -1
    # 使用vectorize函数进行映射
    labels_ = np.vectorize(map_value)(scores)
    # 将映射后的结果转换为NumPy数组
    labels_ = np.array(labels_)
    # print(logits, logits_class)
    print(classification_report(labels_, logits_class))
    return {'4_class_acc':classification_report(labels_, logits_class, output_dict=True)['accuracy']}

"""进行分类，并返回结果
接受的输入为scores:将标注的label映射成的0.15等分数
logits:预测的结果
该函数用于分类函数"""
def metrics_classification(scores, logits,  mode='1'):
    scores = np.array(scores)
    label_dict = {0.15:0, 0.35:1, 0.65:2, 0.9:3}
    def map_value(value):
        for key, label in label_dict.items():
            if np.isclose(value, key):
                return label
        return -1
    # 使用vectorize函数进行映射
    labels_ = np.vectorize(map_value)(scores)
    # 将映射后的结果转换为NumPy数组
    labels_ = np.array(labels_)
    # print(labels_, logits)
    # print(classification_report(labels_, logits))
    return {'4_class_acc':classification_report(labels_, logits, output_dict=True)['accuracy']}



# 计算auc，第二个metric，用于分类模型。
def metric_auc(scores, logits):
    label_dict = {0.05:0, 0.15:0, 0.35:0, 0.65:1, 0.9:1}
    def map_value(value):
        for key, label in label_dict.items():
            if np.isclose(value, key):
                return label
        return -1
    labels_ = np.vectorize(map_value)(scores)
    from collections import Counter
    # print(Counter(labels_))
    # 将映射后的结果转换为NumPy数组
    labels_ = np.array(labels_)
    # print(labels_)
    auc = roc_auc_score(labels_, logits)
    return {'auc':auc}


if __name__=='__main__':
    a = pd.read_csv('./data/test_scores.csv', header=None)
    a.columns = ['query','good_id', 'label', 'model', 'lgb', 'bert']
    a = a.dropna()
    label_dict = {'0':0.15, '1':0.35, '2':0.65, '3':0.9}
    a = a.astype({'lgb':'float','bert':'float','model':'int'})
    a['label'] = a['label'].map(label_dict)
    a = a[a.label.isin([0.15,0.35,0.65,0.9])]
    a['model'] = a['model']-1
    print(a)
    print(metric_auc(a['label'], a['lgb']))
    print(metric_auc(a['label'], a['bert']))
    # print(metrics_classification(a['label'], a['model']))
    print(metrics_1(a['label'],a['bert']))    