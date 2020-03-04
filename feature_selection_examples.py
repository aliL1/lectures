#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

print(pd.__version__)

import sklearn.feature_selection as fs

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from pprint import pprint
import itertools as it
import random as rnd
import statistics as stat
from dataclasses import dataclass, field


# In[2]:


#load sklearn cacer dataset, make a df
cancer = load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'], ['target']))


# In[3]:


#how it works on a single feature, there are some numpy conversations
#it return 1D np array
fs.mutual_info_classif(df['mean radius'].values.reshape(-1,1), df['target'].values.ravel())


# In[4]:


#iterate over all features
feature_mi_pairs = []
for feature_name in list(cancer.feature_names):
    mi = fs.mutual_info_classif(df[feature_name].values.reshape(-1,1), df['target'].values.ravel())
    feature_mi_pairs.append((feature_name, mi[0]))

    
feature_mi_pairs = sorted(feature_mi_pairs, key=lambda tup: tup[1], reverse=True)
pprint(feature_mi_pairs)


# In[5]:


selected_features = [tpl[0] for tpl in feature_mi_pairs[0:10]]
pprint(selected_features)


# In[6]:


#selected_features = cancer['feature_names']


# In[7]:


len(selected_features)


# In[8]:


#split test train - forget the test data
#X = df[cancer['feature_names']]
X = df[selected_features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25,random_state=42)


# In[9]:


X.head()


# In[10]:


def measure_auc(X, y, estimator, estimator_params):
    """
    Given training data X, y and estimator with candidate params find ROC-AUC
    """
 
    estimator = estimator.set_params(**estimator_params)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scores = []
    for train_index, test_index in skf.split(X, y):
        X_train_piece, X_test_piece = X.iloc[train_index], X.iloc[test_index]
        y_train_piece, y_test_piece = y.iloc[train_index], y.iloc[test_index]
            
        estimator.fit(X_train_piece, y_train_piece)
        
        score = roc_auc_score(y_test_piece, estimator.predict(X_test_piece))
        
        scores.append(score)
            
    #print(scores)
    mean = stat.mean(scores)
    return mean


# In[11]:


#An example to set params, given a dictionary of params
estimator = RandomForestClassifier(random_state=42)
params = {'max_depth': 2, 'n_estimators': 40}
estimator = estimator.set_params(**params)

#measure performance
measure_auc(X_train,y_train, estimator, params)


# In[12]:


estimator


# In[13]:


#example below
def generate_param_space(space, num_trials):
    """
    generates all possible combinations in parameter space
    num_trials: sample size to select from all combinations
    """
    
    #Create proper formatted candidates
    bag = []
    
    #key=paramater
    for key in sorted(space.keys()):
        values = space[key]
        key_val_pairs = []
        for val in values:
            pair = (key,val)
            key_val_pairs.append(pair)
        bag.append(key_val_pairs)

    #all candidates but list of tuples
    all_candidate_solutions = list(it.product(*bag))

    #convert to list of dictionaries -> better representation
    all_candid_sol_as_dict = []
    for candid in all_candidate_solutions:
        candid_solution = {}
        for pair in candid:
            key=pair[0]
            value=pair[1]
            candid_solution[key] = value
        all_candid_sol_as_dict.append(candid_solution)

    total_combinations = len(all_candid_sol_as_dict)
    print("Total number of possible parameter combinations:", total_combinations)

    if num_trials > total_combinations:
        num_trials = total_combinations
        print("num_trials > total_combinations, selecting all:", num_trials)
  
    random_candidates = rnd.sample(all_candid_sol_as_dict, num_trials)
    return random_candidates
#####################################################################################

test_space = {"learning_rate" : [0.05, 0.10, 0.15 ],
              "max_depth" : [ 3, 4],
              "min_child_weight" : [ 1, 3, 5, 7 ] }   

# all possible parameter combinations 3x3x4
generate_param_space(test_space, 3) # randomly select 3 of them. becomes a random search


# In[19]:


test_space = {"max_depth" : [2, 3, 4, 5] , 
              "n_estimators" : [50, 60, 70, 80]}   

param_space = generate_param_space(test_space, num_trials=20)

param_auc_list = []

for params in param_space:
    #print('params:', params)
    
    estimator = RandomForestClassifier(random_state=42).set_params(**params) 
    
    auc = measure_auc(X_train, y_train, estimator, params)
    
    param_auc_list.append((estimator, params, auc))

param_auc_list = sorted(param_auc_list, key=lambda tup: tup[2], reverse=True)
#pprint(param_auc_list)

#the best
print('Best Params:')
pprint(param_auc_list[0])


# In[15]:


#Re-fit all training data
best_params = param_auc_list[0][1]
print(best_params)

best_estimator = RandomForestClassifier(random_state=42).set_params(**best_params)

best_estimator.fit(X_train, y_train)


# In[16]:


#performance on training data
roc_auc_score(y_train, best_estimator.predict(X_train))


# In[17]:


#performance on test data
roc_auc_score(y_test, best_estimator.predict(X_test))


# In[18]:


#all - train 0.990566 vs test 0.951153


# In[ ]:





# In[ ]:




