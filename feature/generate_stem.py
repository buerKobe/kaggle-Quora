#encoding:utf-8
##question去除非a-z/A-Z/0-9的字符并将字母全部小写化##

import pandas as pd
import numpy as np
from sklearn.feature_extraction import text  #文本特征提取
from nltk.stem.porter import PorterStemmer   #英语分词
from nltk.stem.snowball import SnowballStemmer
seed = 1024
np.random.seed(seed)
path = './input/'

train = pd.read_csv(path+"train.csv")
test = pd.read_csv(path+"test.csv")

def stem_str(x,stemmer=SnowballStemmer('english')):
    x = text.re.sub("[^a-zA-Z0-9]"," ", x)  #用空格替换x中不是a-z/A-Z/0-9的字符
    x = (" ").join([stemmer.stem(z) for z in x.split(" ")])
    x = " ".join(x.split())  #以空格将句子中每个英文单词分隔
    return x

porter = PorterStemmer()
snowball = SnowballStemmer('english')

print('Generate porter')

train['question1_porter'] = train['question1'].astype(str).apply(lambda x:stem_str(x.lower(),porter)) #英文单词全部小写
test['question1_porter'] = test['question1'].astype(str).apply(lambda x:stem_str(x.lower(),porter))

train['question2_porter'] = train['question2'].astype(str).apply(lambda x:stem_str(x.lower(),porter))
test['question2_porter'] = test['question2'].astype(str).apply(lambda x:stem_str(x.lower(),porter))

train.to_csv(path+'train_porter.csv')
test.to_csv(path+'test_porter.csv')

