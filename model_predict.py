# -*- coding:utf-8 -*-
import os

from sklearn import metrics

import numpy as np
import pandas as pd
import jieba
import re
from gensim.models import word2vec
import math
from sklearn.externals import joblib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

dataset_path = ('./dataset')
# 原始数据的csv文件
output_text_filename = 'raw_weibo_text.csv'
# 清洗好后的csv文件
output_cln_text_filename = 'clean_weibo_text.csv'


def proc_text(raw_line):
    # 使用正则表达式去除非中文字符
    filter_pattern = re.compile('[^\u4E00-\u9FD5]+')
    chinese_only = filter_pattern.sub('', raw_line)

    words = []
    for word in chinese_only:
        words.append(word)

    return ' '.join(words)
    return featureVec

def split_train_test(text_df, size=0.8):
    """
        分割训练集和测试集
    """
    # 为保证每个类中的数据能在训练集中和测试集中的比例相同，所以需要依次对每个类进行处理
    train_text_df = pd.DataFrame()
    test_text_df = pd.DataFrame()

    labels = [0, 1, 2, 3]
    for label in labels:
        # 找出label的记录
        text_df_w_label = text_df[text_df['labels'] == label]
        # 重新设置索引，保证每个类的记录是从0开始索引，方便之后的拆分
        text_df_w_label = text_df_w_label.reset_index()

        # 默认按80%训练集，20%测试集分割

        # 该类数据的行数
        n_lines = text_df_w_label.shape[0]
        split_line_no = math.floor(n_lines * size)
        text_df_w_label_train = text_df_w_label.iloc[:split_line_no, :]
        text_df_w_label_test = text_df_w_label.iloc[split_line_no:, :]

        # 放入整体训练集，测试集中
        train_text_df = train_text_df.append(text_df_w_label_train)
        test_text_df = test_text_df.append(text_df_w_label_test)

    train_text_df = train_text_df.reset_index()
    test_text_df = test_text_df.reset_index()
    return train_text_df, test_text_df


print('加载处理好的文本数据')
clean_text_df = pd.read_csv(os.path.join(dataset_path, output_cln_text_filename), encoding='utf-8')
train_text_df, test_text_df = split_train_test(clean_text_df)
print('训练集中各类数据的个数：', train_text_df.groupby('labels').size())
print('测试集中各类数据的个数：', test_text_df.groupby('labels').size())

# 特征提取
# 把训练集变成句子列表

train=train_text_df['text']
tf=TfidfVectorizer()
X_train=tf.fit_transform(train)
Y_train = train_text_df['labels']
print(Y_train)


gs=MultinomialNB()
gs.fit(X_train, Y_train)


joblib.dump(gs,'gs.m')


stopwords = [line.rstrip() for line in open('中文停用词库.txt', 'r', encoding='utf-8')]
with open('微博评论/李小璐', 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

    text_series = pd.Series(lines)

# 对文档进行只保留中文处理
text = text_series.map(proc_text)

print(text.head())

# 构建句子列表，并去掉词汇之间的空格
text_list = []
for row in text:
    row = ''.join(str(row).split())
    text_list.append(str(row))

print(text_list)

# 对每一句进行分词
texts = [[word for word in jieba.cut(doc)] for doc in text_list]
print(texts)

# 对每一句进行去除停用词
text2 = []

for text in texts:
    words = []
    for text1 in text:
        if text1 not in stopwords:
            words.append(text1)
    text2.append(words)
print(text2)

clf=joblib.load('gs.m')


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

test=test_text_df['text']
word_vec=tf.transform(test)
print('开始预测')
result=clf.predict(word_vec)
print('预测结束')
r=pd.DataFrame(result)
r.to_csv('r.csv')
