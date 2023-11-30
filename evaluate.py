# 用作测试集infer
import os
os.environ['TRANSFORMERS_CACHE'] = '../.cache'
import torch

from transformers import BertTokenizer, AutoModel
from model.models import RegressionModel
import pandas as pd
from tqdm import tqdm

def get_score(checkpoint_path, infenrence_path, save=False):
    scores = []
    df = pd.read_csv(infenrence_path)
    df.columns = ['id', 'query', 'goods_emb_input']
    emb, query, id = [str(x) for x in list(df['goods_emb_input'])], list(df['query']), list(df['id'])
    emb = [x.replace('[SEP]', '#品牌:', 1) for x in emb]
    emb = [x.replace('[SEP]', '#品类:', 1) for x in emb]
    emb = [x.replace('[SEP]', '#单品:', 1) for x in emb]
    emb = [x.replace('[unused1]', ' ', 1) for x in emb]

    print(emb[:5])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model(model_name_or_path).to(device)
    tokenizer = BertTokenizer.from_pretrained(checkpoint_path)

    # tokenizer.add_special_tokens({'additional_special_tokens':["[/BR]","[/CA]"]})
    # model.resize_token_embeddings(len(tokenizer))
    model = RegressionModel.from_pretrained(checkpoint_path).to(device)
    model.eval()
    # model.load_state_dict(checkpoint)
    batch_size = 2000
    with torch.no_grad():
        for i in tqdm(range(0, len(query), batch_size)):
            emb_slice = emb[i:i+batch_size]
            query_slice = query[i:i+batch_size]
            inputs = tokenizer(query_slice,emb_slice,truncation=True, padding=True, 
                                        return_token_type_ids=True, return_tensors="pt", max_length=100).to(device)    
            score = model(input_ids=inputs['input_ids'], 
                          attention_mask=inputs['attention_mask'], 
                          token_type_ids=inputs['token_type_ids'])
            score = score.cpu().flatten().tolist()
            scores.extend(score)
    new_df = pd.DataFrame({'id':id, 'query':query, 'score':scores})
    return new_df



def evaluate(checkpoint_path, infenrence_path, delimiter ='\t'):
    scores = []
    df = pd.read_csv(infenrence_path, sep=delimiter)
    # df.columns = ['id', 'query','score', 'goods_emb_input']
    emb, query = [str(x) for x in list(df['sentence1'])], list(df['sentence2'])

   #  print(emb[:5])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model(model_name_or_path).to(device)
    tokenizer = BertTokenizer.from_pretrained(checkpoint_path)

    # tokenizer.add_special_tokens({'additional_special_tokens':["[/BR]","[/CA]"]})
    # model.resize_token_embeddings(len(tokenizer))
    model = RegressionModel.from_pretrained(checkpoint_path).to(device)
    model = model.to(device)
    model.eval()
    # model.load_state_dict(checkpoint)
    batch_size = 500
    with torch.no_grad():
        for i in tqdm(range(0, len(query), batch_size)):
            emb_slice = emb[i:i+batch_size]
            query_slice = query[i:i+batch_size]
            emb_slice = [x[:15] for x in emb_slice]
            inputs = tokenizer(emb_slice,
                               query_slice,
                               truncation=True,
                               padding='max_length',
                               return_token_type_ids=True,
                               return_tensors="pt",
                               max_length=450).to(device)
            score = model(input_ids=inputs['input_ids'], 
                          attention_mask=inputs['attention_mask'], 
                          token_type_ids=inputs['token_type_ids']
                          )
            score = score.cpu().flatten().tolist()
            scores.extend(score)
    new_df = pd.DataFrame({ 'sentence1':query, 'pred':scores, 'score':df['label'], 'sentence2':emb})
    score_mapping = {0.05:0, 0.15:0, 0.35:0, 0.65:1, 0.9:1}
    print(new_df['score'])
    print(score_mapping)
    # # Map the 'score' column using the score_mapping dictionary
    new_df['label'] = new_df['score'].map(score_mapping)
    from sklearn.metrics import roc_auc_score
   #  print(new_df)
    auc_score = roc_auc_score(new_df['label'], new_df['pred'])
    print(infenrence_path,'*'*50)
    print('auc score is:', auc_score)

    return new_df

def diff_rate(score1, score2):
    # print(score1.columns)
    score1.columns = ['id','query','score1', 'label']
    score2.columns = ['id','query','score2']
    compare = score1.merge(score2, how='inner', on=['id','query'])
    compare = compare.dropna()
    score_1 = list(compare['score1'])
    score_2 = list(compare['score2'])
    if len(score_1)!=len(score_2):
        print('wrong!')
    count = 0
    def threshold(num):
        if num<0.25:
            return 0
        if num<0.5:
            return 1
        if num<0.75:
            return 2
        return 3
    for i in range(len(score_1)):
        score_1[i] = threshold(score_1[i])
        score_2[i] = threshold(score_2[i])
    for i in range(len(score_1)):
        if score_1[i]!=score_2[i]:
            count+=1
        # print(score_1[i],score_2[i])
    from collections import Counter
    print('Counter for 1:', Counter(score_1))
    print('Counter for 2:', Counter(score_2))
    print('Diff率为:', count/len(score_1))


def pos_neg_rate(score1, score2, is_full=False):
    score1.columns = ['id','query','score1', 'label']
    score2.columns = ['id','query','score2']
    compare = score1.merge(score2, how='inner', on=['id','query'])
    compare = compare.dropna()
    score_1 = list(compare['score1'])
    score_2 = list(compare['score2'])
    label = list(compare['label'])
    def threshold(num):
        if num<0.25:
            return 0
        if num<0.5:
            return 1
        if num<0.75:
            return 2
        return 3
    for i in range(len(score_1)):
        score_1[i] = threshold(score_1[i])
        score_2[i] = threshold(score_2[i])

    count1, count2 = 0,0
    neg1, neg2 = 0, 0
    for i in range(len(score1)):
        if label[i]!=3:
            continue

        if score_1[i]==3:
            count1+=1
        else:
            neg1+=1
        
        if score_2[i]==3:
            count2+=1
        else:
            neg2+=1
    
    print(count1/neg1)
    print(count2/neg2)
    # import random
    # indexs = [x for x in [i for i in range(len(score1))] if random.random()<0.1]
    # score_1 = [score_1[i] for i in range(len(score_1)) if i in indexs]
    # score_2 = [score_2[i] for i in range(len(score_2)) if i in indexs]
    # label = [label[i] for i in range(len(label)) if i in indexs]
    # for i in tqdm(range(len(score_1))):
    #     for j in range(len(score_2)):
    #         if score_1[i]>score_1[j] and label[i]<label[j]:
    #             count_1+=1
    #         if score_2[i]>score_2[j] and label[i]<label[j]:
    #             count_2+=1
    

    # print('Reverse of 1:%d'%count_1)
    # print('Reverse of 2:%d'%count_2)


if __name__=='__main__':
   model_path = '/mnt/search01/usr/laodanning/GoodnotesTeacher/outputs/teacher_distill_base/checkpoint-24000'
   data_path = '/mnt/search01/dataset/laodanning/data/good_notes/eval/'
   score_3 = evaluate(model_path, data_path+'/random_easy',)
   # # 0.8190870870521543
   # score_4 = evaluate(model_path,
   #                   data_path+'/random_hard',)
   #  0.669356126731072
#    score_5 = evaluate(model_path, "../datasets/notes/v1/train.csv",delimiter=',')
   df = score_3
   df.to_csv("/mnt/search01/usr/laodanning/GoodnotesTeacher/outputs/eval/1130.csv", index=False, encoding="utf-8-sig")