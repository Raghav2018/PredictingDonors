# -*- coding: utf-8 -*-
"""
Created on Sun May 07 21:34:16 2017

@author: raghavendra harish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from Helper import Preprocessor as Preprocessor
from Helper import Performance as Performance

raw_data = pd.read_csv("cup98LRN.txt",sep = ',',
            error_bad_lines = False, low_memory = False,
            skip_blank_lines = True, na_values = [' '],
            keep_default_na = True, verbose = True)

# Exploratory analysis
    
raw_data.shape
raw_data.count()
raw_data.head()
raw_data.columns

# Lets the distribution of the target variables
plt.plot(raw_data.TARGET_B)

# Correlation between TARGET_B and the predictors
TARGET_B_corr = raw_data.corr()["TARGET_B"].copy()
TARGET_B_corr.sort(ascending = False)
TARGET_B_corr

# Some statistics about raw_data's variables
raw_data.describe()

# General description of variables

#[1:28] # demographics
#[29:42] # response to other types of mail orders
#[43:55] # overlay data
#[56:74] # donor interests
#[75] # PEP star RFA status
#[76:361] # characteristics of donor neighborhood
#[362:407] # promotion history
#[408:412] # summary variables of promotion history
#[413:456] # giving history
#[457:469] # summary variables of giving history
#[470:473] ## ID & TARGETS
#[474:479] # RFA (recency-frequency-donation amount)
#[480:481] # cluster & geocode

print 'Donor Percentage: %s' % (
        100.0 * sum(raw_data.TARGET_B) / raw_data.shape[0])

#Data Cleaning - Higly unclean data with high dimension and lot of 
#missing values and only ~5% positive cases.

#rounding up all removable variables 

#removing some variables based on business intuition. 
removable_vars = ['CONTROLN', 'ZIP', 'DOB']

#removing variables that are strongly correlated
data_correlation = raw_data.corr()
data_correlation.loc[:, :] = np.tril(data_correlation, k = -1)
data_correlation = data_correlation.stack()

# Get list of correlated vars greater than 0.1
corr_pairs = data_correlation[data_correlation > 0.1].to_dict().keys()

chosen_vars = [i[0] for i in corr_pairs]
chosen_vars.extend([i[1] for i in corr_pairs if i[1] not in chosen_vars])

redundant_vars = [var for var in [
            x for t in corr_pairs for x in t] if var not in chosen_vars]

removable_vars.extend(redundant_vars)

#removing variables that are very sparse
idxs = raw_data.count() < int(raw_data.shape[0] * .01)
removable_vars.extend(raw_data.columns[idxs])

#Lets drop them all 
semi_clean_data = raw_data.drop(removable_vars, axis =1)

# changing some variabes to be treated as categorical
for col in ['ODATEDW', 'OSOURCE', 'TCODE', 'STATE', 'MAILCODE', 
'PVASTATE', 'NOEXCH', 'RECINHSE', 'RECP3', 'RECSWEEP', 'MDMAUD', 
'DOMAIN', 'AGEFLAG', 'HOMEOWNR', 'CHILD03', 'CHILD07', 'CHILD12', 
'CHILD18', 'NUMCHLD', 'INCOME', 'GENDER', 'WEALTH1', 'MBCRAFT', 
'MBGARDEN', 'MBBOOKS', 'MBCOLECT', 'MAGFAML', 'MAGFEM', 'MAGMALE', 
'PUBGARDN', 'PUBCULIN', 'PUBHLTH', 'PUBDOITY', 'PUBNEWFN', 'PUBPHOTO', 
'PUBOPP', 'DATASRCE', 'SOLIH', 'WEALTH2', 'GEOCODE', 'COLLECT1', 
'VETERANS', 'BIBLE', 'CATLG', 'PETS', 'CDPLAY', 'STEREO', 
'PCOWNERS', 'PHOTO', 'CRAFTS', 'FISHER', 'GARDENIN', 'BOATS', 
'WALKER', 'KIDSTUFF', 'CARDS', 'LIFESRC', 'PEPSTRFL', 'ADATE_4', 
'ADATE_6', 'ADATE_7', 'ADATE_8', 'ADATE_9', 'ADATE_10', 'ADATE_11', 
'ADATE_12', 'ADATE_13', 'ADATE_14', 'ADATE_16', 'ADATE_17', 'ADATE_18', 
'ADATE_19', 'ADATE_21', 'ADATE_22', 'ADATE_23', 'ADATE_24', 'RFA_2', 
'RFA_3', 'RFA_4', 'RFA_5', 'RFA_6', 'RFA_7', 'RFA_8', 'RFA_9', 
'RFA_10', 'RFA_11', 'RFA_12', 'RFA_13', 'RFA_14', 'RFA_15', 
'RFA_16', 'RFA_17', 'RFA_18', 'RFA_19', 'RFA_20', 'RFA_21', 
'RFA_22', 'RFA_23', 'RFA_24', 'MAXADATE', 'NUMPROM', 
'CARDPM12', 'RDATE_7', 'RDATE_8', 'RDATE_9', 'RDATE_10', 
'RDATE_11', 'RDATE_12', 'RDATE_13', 'RDATE_14', 'RDATE_15', 
'RDATE_16', 'RDATE_17', 'RDATE_18', 'RDATE_19', 'RDATE_20', 
'RDATE_21', 'RDATE_22', 'RDATE_23', 'RDATE_24', 'RAMNT_7', 
'RAMNT_8', 'RAMNT_9', 'RAMNT_10', 'RAMNT_11', 'RAMNT_12', 
'RAMNT_13', 'RAMNT_14', 'RAMNT_15', 'RAMNT_16', 
'RAMNT_17', 'RAMNT_18', 'RAMNT_19', 'RAMNT_20', 'RAMNT_21', 
'RAMNT_22', 'RAMNT_23', 'RAMNT_24','MINRDATE', 'MAXRAMNT', 
'MAXRDATE', 'LASTGIFT', 'LASTDATE', 'FISTDATE', 'NEXTDATE', 
'HPHONE_D', 'RFA_2R', 'RFA_2F', 'RFA_2A', 'MDMAUD_R', 
'MDMAUD_F', 'MDMAUD_A', 'GEOCODE2']:
        semi_clean_data[col] = semi_clean_data[col].astype(object)

# handling missing values, replaced by mean/frequent value  
clean_data = Preprocessor.fill_nans(semi_clean_data)

#Lets do some feature selection to improve perfromance
#Using correlation based method

idxs_pos = clean_data['TARGET_B'] == 1
pos = clean_data[idxs_pos]
neg = clean_data[clean_data['TARGET_B'] == 0][1:sum(idxs_pos)]
sub_dat = pos.append(neg, ignore_index = True)
sub_dat = Preprocessor.fill_nans(sub_dat)
X = pd.get_dummies(sub_dat)

# Computing correlation between 'TARGET_B' and the predictors
target_corr = X.corr()['TARGET_B'].copy()
target_corr.sort(ascending = False)

# picking up the top important feature      
tmp = abs(target_corr).copy()
tmp.sort(ascending = False)
important_vars = [tmp.index[0]]
important_vars.extend(list(tmp.index[2:52]))

feats = pd.get_dummies(clean_data)

# Lets drop the unimportant variables
feats = feats[important_vars]

# Lets split data for training and verification, 70% and 30% respectively
cut = int(feats.shape[0] * .7)

train = feats[1:cut].drop(['TARGET_B'], axis = 1)
y_train = feats.TARGET_B[1:cut]

test = feats[(cut + 1):-1].drop(['TARGET_B'], axis = 1)
y_test = feats.TARGET_B[(cut + 1):-1]

# Creating a balanced trainset to improve performance
pos = train[y_train == 1]
neg = train[y_train == 0][1:pos.shape[0]]
y_train_bal = [1] * pos.shape[0]
y_train_bal.extend([0] * neg.shape[0])
train_bal = pos.append(neg, ignore_index = True)

# Model Training and Evaluation #

#### Training ####
#### Model 1 | Decision Tree Model ####

print "Model 1 executing..."

# Training
clf = DecisionTreeClassifier(max_depth = 20) # TODO: should let the tree fully grow
# and then prune it automatically according to an optimal depth
clf = clf.fit(train_bal.values, y_train_bal)

# Testing
y_test_pred = clf.predict(test.values)
y_all_models = y_test_pred.copy()

# Confusion Matrix
print pd.crosstab(
        y_test, y_test_pred, rownames = ['actual'], colnames = ['preds'])

# Gets performance
perf_model1 = Performance.get_perf(y_test.values, y_test_pred)

#### Model 2 | Random Forest Model ####

print "Model 2 executing..."

# Training
clf = ExtraTreesClassifier(n_estimators = 500, verbose = 1,
        bootstrap = True, max_depth = 20, oob_score = True, n_jobs = -1)

#clf = RandomForestClassifier(
#    n_estimators = 500, max_depth = 10, verbose = 1, n_jobs = -1)

clf = clf.fit(train_bal.values, y_train_bal)
    
X = train_bal.values
# Testing
y_test_pred = clf.predict(test.values)
y_all_models += y_test_pred

# Confusion Matrix
print pd.crosstab(
        y_test, y_test_pred, rownames = ['actual'], colnames = ['preds'])
    
#importances = clf.feature_importances_
#    
#std = np.std([tree.feature_importances_ for tree in clf.estimators_],
#             axis=0)
#indices = np.argsort(importances)[::-1]
#    
## Print the feature ranking
#print("Feature ranking:")
#    
#for f in range(X.shape[1]):
#    print f
#print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#    
## Plot the feature importances of the forest
#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(X.shape[1]), importances[indices],
#           color="r", yerr=std[indices], align="center")
#plt.xticks(range(X.shape[1]), indices)
#plt.xlim([-1, X.shape[1]])
#plt.show()
#
#print importances

# Gets performance
perf_model2 = Performance.get_perf(y_test, y_test_pred)

#### Model 3 | Logistic Regression Model ####

print "Model 3 executing..."

# Training
clf = LogisticRegression(max_iter = 200, verbose = 1)
clf = clf.fit(train_bal.values, y_train_bal)

# Testing
y_test_pred = clf.predict(test.values)
y_all_models += y_test_pred

# Confusion Matrix
print pd.crosstab(
        y_test, y_test_pred, rownames = ['actual'], colnames = ['preds'])

# Gets performance
perf_model3 = Performance.get_perf(y_test, y_test_pred)

#### Model 4 | Ensemble Model (majority vote for model 1, 2 and 3) ####

print "Model 4 executing..."

# Gets performance for an ensemble of all 3 models
y_test_pred = np.array([0] * len(y_all_models))
y_test_pred[y_all_models > 1] = 1
perf_model_ensemble = Performance.get_perf(y_test, y_test_pred)

# Confusion Matrix
print pd.crosstab(
        y_test, y_test_pred, rownames = ['actual'], colnames = ['preds'])

#### Model comparison ####

all_models = {'Decision Trees Model': perf_model1,
                  'Random Forest Model': perf_model2,
                  'Logistic Regression Model': perf_model3,
                  'Ensemble Model': perf_model_ensemble}

perf_all_models = pd.DataFrame([[col1, col2, col3 * 100] for col1, d in
        all_models.items() for col2, col3 in d.items()], index = None,
        columns = ['Model Name', 'Performance Metric', 'Value'])

print perf_all_models







