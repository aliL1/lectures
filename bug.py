import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.feature_selection as fs

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer


from pprint import pprint
import itertools as it
import random as rnd
import statistics as stat


cancer = load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'], ['target']))

y_train = df['target']
X_train = df.drop(['target'] , axis = 1)


clf =  LogisticRegression(C=0.1, max_iter=4000, random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=skf)
print(scores) #[0.74801587 0.63690476 0.66112266 0.66112266 0.72349272]

scores = []
for train_index, test_index in skf.split(X_train, y_train):
    X_train_p, X_test_p = X_train.loc[train_index], X_train.loc[test_index]
    y_train_p, y_test_p = y_train.loc[train_index], y_train.loc[test_index]
    
    clf.fit(X_train_p, y_train_p)
    
    score = roc_auc_score(y_test_p, clf.predict_proba(X_test_p)[:,1])
        
    scores.append(score)

print(scores)#[0.748015873015873, 0.636904761904762, 0.6611226611226612, 0.661122661122661, 0.7234927234927235]

#found a bug
scores = cross_val_score(clf, X_train, y_train, scoring=make_scorer(roc_auc_score), cv=skf)
print(scores)#[0.5734127  0.5734127  0.5956341  0.66320166 0.58627859]
print('last piece is a bug')
