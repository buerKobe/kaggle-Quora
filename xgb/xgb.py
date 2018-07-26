#encoding:utf8
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.model_selection import KFold
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.utils import resample,shuffle
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from math import sqrt

seed=1024
NFOLDS = 5
SEED = 0
NROWS = None
np.random.seed(seed)
path = "../input/"
train = pd.read_csv(path+"train_porter.csv")
SUBMISSION_FILE = '../input/sample_submission.csv'


# 读入特征
train_question1_tfidf = pd.read_pickle(path+'train_question1_tfidf.pkl')[:]
test_question1_tfidf = pd.read_pickle(path+'test_question1_tfidf.pkl')[:]

train_question2_tfidf = pd.read_pickle(path+'train_question2_tfidf.pkl')[:]
test_question2_tfidf = pd.read_pickle(path+'test_question2_tfidf.pkl')[:]

train_question1_porter_tfidf = pd.read_pickle(path+'train_question1_porter_tfidf.pkl')[:]
test_question1_porter_tfidf = pd.read_pickle(path+'test_question1_porter_tfidf.pkl')[:]

train_question2_porter_tfidf = pd.read_pickle(path+'train_question2_porter_tfidf.pkl')[:]
test_question2_porter_tfidf = pd.read_pickle(path+'test_question2_porter_tfidf.pkl')[:]

train_interaction = pd.read_pickle(path+'train_interaction.pkl')[:].reshape(-1,1)
test_interaction = pd.read_pickle(path+'test_interaction.pkl')[:].reshape(-1,1)

train_porter_interaction = pd.read_pickle(path+'train_porter_interaction.pkl')[:].reshape(-1,1)
test_porter_interaction = pd.read_pickle(path+'test_porter_interaction.pkl')[:].reshape(-1,1)


train_jaccard = pd.read_pickle(path+'train_jaccard.pkl')[:].reshape(-1,1)
test_jaccard = pd.read_pickle(path+'test_jaccard.pkl')[:].reshape(-1,1)

train_porter_jaccard = pd.read_pickle(path+'train_porter_jaccard.pkl')[:].reshape(-1,1)
test_porter_jaccard = pd.read_pickle(path+'test_porter_jaccard.pkl')[:].reshape(-1,1)

train_len = pd.read_pickle(path+"train_len.pkl")[:]
test_len = pd.read_pickle(path+"test_len.pkl")[:]

train_q1_freq = pd.read_pickle(path+"train_q1_freq.pkl")[:]
test_q1_freq = pd.read_pickle(path+"test_q1_freq.pkl")[:]

train_q2_freq = pd.read_pickle(path+"train_q2_freq.pkl")[:]
test_q2_freq = pd.read_pickle(path+"test_q2_freq.pkl")[:]

train_q1_hash = pd.read_pickle(path+"train_q1_hash.pkl")[:]
test_q1_hash = pd.read_pickle(path+"test_q1_hash.pkl")[:]

train_q2_hash = pd.read_pickle(path+"train_q2_hash.pkl")[:]
test_q2_hash = pd.read_pickle(path+"test_q2_hash.pkl")[:]

train_q1_q2_intersect = pd.read_pickle(path+"train_q1_q2_intersect.pkl")[:]
test_q1_q2_intersect = pd.read_pickle(path+"test_q1_q2_intersect.pkl")[:]

scaler = MinMaxScaler()
scaler.fit(np.vstack([train_len,test_len]))
train_len = scaler.transform(train_len)
test_len = scaler.transform(test_len)

X = ssp.hstack([
    train_question1_tfidf,
    train_question2_tfidf,
    train_interaction,
    train_porter_interaction,
    train_jaccard,
    train_porter_jaccard,
    train_len,
    train_q1_freq,
	train_q2_freq,
	train_q1_hash,
	train_q2_hash,
	# train_q1_q2_intersect,
    ]).tocsr()

y = train['is_duplicate'].values[:]

X_t = ssp.hstack([
    test_question1_tfidf,
    test_question2_tfidf,
    test_interaction,
    test_porter_interaction,
    test_jaccard,
    test_porter_jaccard,
    test_len,
	test_q1_freq,
	test_q2_freq,
	test_q1_hash,
	test_q2_hash,
	# test_q1_q2_intersect,
    ]).tocsr()

skf = KFold(n_splits=5, shuffle=True, random_state=seed).split(X)
for ind_tr, ind_te in skf:
    X_train = X[ind_tr]
    X_test = X[ind_te]

    y_train = y[ind_tr]
    y_test = y[ind_te]
    break

def oversample(X_ot,y,p=0.165):
    pos_ot = X_ot[y==1]
    neg_ot = X_ot[y==0]
    #p = 0.165
    scale = ((pos_ot.shape[0]*1.0 / (pos_ot.shape[0] + neg_ot.shape[0])) / p) - 1
    while scale > 1:
        neg_ot = ssp.vstack([neg_ot, neg_ot]).tocsr()
        scale -=1
    neg_ot = ssp.vstack([neg_ot, neg_ot[:int(scale * neg_ot.shape[0])]]).tocsr()
    ot = ssp.vstack([pos_ot, neg_ot]).tocsr()
    y=np.zeros(ot.shape[0])
    y[:pos_ot.shape[0]]=1.0
    # print y.mean()
    return ot,y

X_train,y_train = oversample(X_train.tocsr(),y_train,p=0.165)
X_train,y_train = shuffle(X_train,y_train,random_state=seed)

dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_t)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01, #0.01
    'objective': 'binary:logistic',  #'reg:linear'
    'max_depth': 7, #1
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'logloss', #'error'
}

start_time = datetime.now()
res = xgb.cv(xgb_params, dtrain, num_boost_round=2500, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=50, verbose_eval=500, show_stdv=True)
			 
best_nrounds = res.shape[0] - 1  #得到最理想的决策树数量
print best_nrounds
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)
submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = gbdt.predict(dtest)
submission.to_csv('xgb.csv', index=None)
end_time = datetime.now()
duration = end_time - start_time
print ('time: %s'%duration)

