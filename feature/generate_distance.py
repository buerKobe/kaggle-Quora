#encoding:utf-8
##question杰卡德相似距离##

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import distance
seed = 1024
np.random.seed(seed)
path = "./input/"

train = pd.read_csv(path+"train_porter.csv")
test = pd.read_csv(path+"test_porter.csv")
test['is_duplicated']=[-1]*test.shape[0]
# print test['is_duplicated']
len_train = train.shape[0]
# print len_train
data_all = pd.concat([train,test])
# print data_all
def str_jaccard(str1, str2): #杰卡德相似距离


    str1_list = str1.split(" ")
    str2_list = str2.split(" ")
    res = distance.jaccard(str1_list, str2_list)
    return res

def str_levenshtein_1(str1, str2): # shortest alignment

    str1_list = str1.split(' ')
    str2_list = str2.split(' ')
    res = distance.nlevenshtein(str1_list, str2_list,method=1)
    return res

def str_levenshtein_2(str1, str2): # longest alignment

    str1_list = str1.split(' ')
    str2_list = str2.split(' ')
    res = distance.nlevenshtein(str1_list, str2_list,method=2)
    return res

def str_sorensen(str1, str2):

    str1_list = str1.split(' ')
    str2_list = str2.split(' ')
    res = distance.sorensen(str1_list, str2_list)
    return res

print('Generate jaccard')
train_jaccard = train.astype(str).apply(lambda x:str_jaccard(x['question1'],x['question2']),axis=1)
test_jaccard = test.astype(str).apply(lambda x:str_jaccard(x['question1'],x['question2']),axis=1)
# print train_jaccard
# print test_jaccard
pd.to_pickle(train_jaccard,path+"train_jaccard.pkl")
pd.to_pickle(test_jaccard,path+"test_jaccard.pkl")

print('Generate porter jaccard')
train_porter_jaccard = train.astype(str).apply(lambda x:str_jaccard(x['question1_porter'],x['question2_porter']),axis=1)
test_porter_jaccard = test.astype(str).apply(lambda x:str_jaccard(x['question1_porter'],x['question2_porter']),axis=1)
# print train_jaccard
# print test_jaccard
pd.to_pickle(train_porter_jaccard,path+"train_porter_jaccard.pkl")
pd.to_pickle(test_porter_jaccard,path+"test_porter_jaccard.pkl")

#其他距离
# print('Generate levenshtein_1')
# train_levenshtein_1 = train.astype(str).apply(lambda x:str_levenshtein_1(x['question1'],x['question2']),axis=1)
# test_levenshtein_1 = test.astype(str).apply(lambda x:str_levenshtein_1(x['question1'],x['question2']),axis=1)
# pd.to_pickle(train_levenshtein_1,path+"train_levenshtein_1.pkl")
# pd.to_pickle(test_levenshtein_1,path+"test_levenshtein_1.pkl")

# print('Generate porter levenshtein_1')
# train_porter_levenshtein_1 = train.astype(str).apply(lambda x:str_levenshtein_1(x['question1_porter'],x['question2_porter']),axis=1)
# test_porter_levenshtein_1 = test.astype(str).apply(lambda x:str_levenshtein_1(x['question1_porter'],x['question2_porter']),axis=1)
# pd.to_pickle(train_porter_levenshtein_1,path+"train_porter_levenshtein_1.pkl")
# pd.to_pickle(test_porter_levenshtein_1,path+"test_porter_levenshtein_1.pkl")

# print('Generate levenshtein_2')
# train_levenshtein_2 = train.astype(str).apply(lambda x:str_levenshtein_2(x['question1'],x['question2']),axis=1)
# test_levenshtein_2 = test.astype(str).apply(lambda x:str_levenshtein_2(x['question1'],x['question2']),axis=1)
# pd.to_pickle(train_levenshtein_2,path+"train_levenshtein_2.pkl")
# pd.to_pickle(test_levenshtein_2,path+"test_levenshtein_2.pkl")

# print('Generate porter levenshtein_2')
# train_porter_levenshtein_2 = train.astype(str).apply(lambda x:str_levenshtein_2(x['question1_porter'],x['question2_porter']),axis=1)
# test_porter_levenshtein_2 = test.astype(str).apply(lambda x:str_levenshtein_2(x['question1_porter'],x['question2_porter']),axis=1)
# pd.to_pickle(train_porter_levenshtein_2,path+"train_porter_levenshtein_2.pkl")
# pd.to_pickle(test_porter_levenshtein_2,path+"test_porter_levenshtein_2.pkl")

# print('Generate sorensen')
# train_sorensen = train.astype(str).apply(lambda x:str_sorensen(x['question1'],x['question2']),axis=1)
# test_sorensen = test.astype(str).apply(lambda x:str_sorensen(x['question1'],x['question2']),axis=1)
# pd.to_pickle(train_sorensen,path+"train_sorensen.pkl")
# pd.to_pickle(test_sorensen,path+"test_sorensen.pkl")

# print('Generate porter sorensen')
# train_porter_sorensen = train.astype(str).apply(lambda x:str_sorensen(x['question1_porter'],x['question2_porter']),axis=1)
# test_porter_sorensen = test.astype(str).apply(lambda x:str_sorensen(x['question1_porter'],x['question2_porter']),axis=1)
# pd.to_pickle(train_porter_sorensen,path+"train_porter_sorensen.pkl")
# pd.to_pickle(test_porter_sorensen,path+"test_porter_sorensen.pkl")


