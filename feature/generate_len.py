#encoding:utf-8
##移除停止词求得各类长度值词干##

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import distance
from nltk.corpus import stopwords
import nltk
seed = 1024
np.random.seed(seed)
path = './input/'

train = pd.read_csv(path+"train_porter.csv").astype(str)
test = pd.read_csv(path+"test_porter.csv").astype(str)


def str_abs_diff_len(str1, str2): #字符串长度之差
    return abs(len(str1)-len(str2))

def str_len(str1): #字符串长度
    return len(str(str1))

def char_len(str1): #字符串净长度(不含空格)
    str1_list = set(str(str1).replace(' ',''))
    return len(str1_list)

def word_len(str1): #单词数
    str1_list = str1.split(' ')
    return len(str1_list)

stop_words = stopwords.words('english')  #英文停止词对英语句子辨析无作用，不予考虑
def word_match_share(row): #除去停止词后共同单词长度的比例
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stop_words:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stop_words:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # 会有一些只有stopwords的question
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words] 
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))*1.0/(len(q1words) + len(q2words))
    return R

print('Generate len')
feats = []
#question字符数之差
train['abs_diff_len'] = train.apply(lambda x:str_abs_diff_len(x['question1'],x['question2']),axis=1)
test['abs_diff_len']= test.apply(lambda x:str_abs_diff_len(x['question1'],x['question2']),axis=1)
# print train['abs_diff_len']
feats.append('abs_diff_len')
#question长度相似率(不包含stopwords)
train['R']=train.apply(word_match_share, axis=1, raw=True)
test['R']=test.apply(word_match_share, axis=1, raw=True)
# print train['R']
feats.append('R')
#question相同单词的个数
train['common_words'] = train.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
test['common_words'] = test.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
# print train['common_words'] 
feats.append('common_words')

for c in ['question1','question2']:
    train['%s_char_len'%c] = train[c].apply(lambda x:char_len(x))
    test['%s_char_len'%c] = test[c].apply(lambda x:char_len(x))
    # print train['%s_char_len'%c]
    feats.append('%s_char_len'%c)

    train['%s_str_len'%c] = train[c].apply(lambda x:str_len(x))
    test['%s_str_len'%c] = test[c].apply(lambda x:str_len(x))
    # print train['%s_str_len'%c]
    feats.append('%s_str_len'%c)
    
    train['%s_word_len'%c] = train[c].apply(lambda x:word_len(x))
    # print train['%s_word_len'%c]
    test['%s_word_len'%c] = test[c].apply(lambda x:word_len(x))

    feats.append('%s_word_len'%c)
# print  feats
train=train[feats].values
print train.shape
print train[feats].values
print train[feats].values[1]
pd.to_pickle(train[feats].values,path+"train_len.pkl")
pd.to_pickle(test[feats].values,path+"test_len.pkl")
