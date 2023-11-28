import pandas as pd
from datasets import Dataset


def remove_duplicate_in_train(train_dataset, eval_dataset):
    train_df, eval_df = train_dataset.to_pandas(), eval_dataset.to_pandas()
    # print(train_dataset,eval_dataset)
    print('before removing:%d'%len(train_df))
    # 使用布尔索引筛选出在 eval_dataset 中 'id' 和 'search_word' 列同时与 train_dataset 重合的部分的行的索引
    if 'goods_id' in train_df.columns:
        on = ['goods_id','search_word']
    else:
        on = ['goods_emb_input', 'search_word']
    train_dataset_deduplicated=train_df.merge(eval_df[on], on=on, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', 1)
    # 在 train_dataset 中删除与 eval_dataset 重合的行

    if len(train_dataset_deduplicated)==len(train_df):
        print("No duplicated item!")
    else:
        print("%d duplicated item deleted."%(len(train_df)-len(train_dataset_deduplicated)))

    return Dataset.from_pandas(train_dataset_deduplicated, preserve_index=False), eval_dataset


def remove_duplicate_in_train_df(train_dataset, eval_dataset):
    
    def process_row(row):
        row['goods_emb_input'] = row['goods_emb_input'].replace('[SEP]', '#品牌:', 1)
        row['goods_emb_input'] = row['goods_emb_input'].replace('[SEP]', '#品类:', 1)
        row['goods_emb_input'] = row['goods_emb_input'].replace('[SEP]', '#单品:', 1)
        row['goods_emb_input'] = row['goods_emb_input'].replace('[unused1]', ' ', 1)
        row['goods_emb_input'] = row['goods_emb_input'].replace('[unused2]', ' ', 1)
        return row
    eval_dataset = eval_dataset.apply(process_row, axis=1)
    # Apply the process_row function to each row of the DataFrame
    print(eval_dataset)
    train_df, eval_df = train_dataset, eval_dataset
    # print(train_dataset,eval_dataset)
    print('before removing:%d'%len(train_df))
    # 使用布尔索引筛选出在 eval_dataset 中 'id' 和 'search_word' 列同时与 train_dataset 重合的部分的行的索引
    if 'goods_id' in train_df.columns:
        on = ['goods_id','search_word']
    else:
        on = ['goods_emb_input', 'search_word']
    train_dataset_deduplicated=train_df.merge(eval_df[on], on=on, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', 1)
    # 在 train_dataset 中删除与 eval_dataset 重合的行

    if len(train_dataset_deduplicated)==len(train_df):
        print("No duplicated item!")
    else:
        print("%d duplicated item deleted."%(len(train_df)-len(train_dataset_deduplicated)))

    return train_dataset_deduplicated, eval_df



def clean_training_data(df):
    def checker(example):
        if int(example['score']) in [-1,0]:
            if example['teacher']>0.5:
                return False
            else:
                return True
        elif int(example['score']) in [1,2]:
            if example['teacher']<0.5:
                return False
            else:
                return True
        return False
    print(df)
    print(df.columns)
    print(df.apply(checker, axis=1))
    df = df[df.apply(checker, axis=1)]
    return df


if __name__=='__main__':
    train_df = pd.read_csv('~/datasets/posttrain_v2_teacher.csv')
    test_df = pd.read_csv('./data/train/test.csv')
    train_df = clean_training_data(train_df)
    train_df, test_df = remove_duplicate_in_train_df(train_df, test_df)
    train_df.to_csv('~/datasets/posttrain_v2_teacher_cleaned.csv', index=False)