#!/usr/bin/env python
# coding: utf-8

# # Libraries to import

# In[10]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# # Generic functions to get the baseline models, ie with the default parameters

# Function 1: Let's define a function which will instantiate as many models 
#as you wish to

def Reg_GetBasedModel():
    basedModels = []
    basedModels.append(('LinearR'   , LinearRegression()))
    basedModels.append(('Ridge'   , Ridge()))     
    basedModels.append(('Lasso'   , Lasso()))
    basedModels.append(('DT-R' , DecisionTreeRegressor()))
    basedModels.append(('SVM-R'  , SVR()))
    basedModels.append(('RF-R'   , RandomForestRegressor()))
    return basedModels


# In[12]:


# Function 2: Let's define a function that will train 
# each individual model described in GetBasedModel() function

def Reg_basedModels(X_train, y_train,scoring, models):
    """
    BasedModels will return the evaluation metric after performing
    a CV for each of the models
    input:
    X_train
    y_train
    models = array containing the different instantiated models
    
    output:
    names = names of the diff models tested
    results = results of the diff models
    """
    # Test options and evaluation metric
    num_folds = 10
    scoring = scoring

    results = []
    names = []
    
    for name, model in models:
        cv_results = cross_val_score(model, X_train,
                                     y_train, cv=num_folds, scoring=scoring)
        results.append(cv_results.mean())
        names.append(name)
        msg = "%s: %s = %f (std = %f)" % (name, scoring,
                                                cv_results.mean(), 
                                                cv_results.std())
        print(msg)
    scoreDataFrame = pd.DataFrame({'Model':names, 'Score': results})
       
        
    return scoreDataFrame




# # Pipeline with the scaling methods

# In[14]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# We will define a function to apply any preprocessing method to the raw data

def Reg_GetScaledModel(nameOfScaler):
    """
    Function to define whether we want to apply any preprocessing method to the raw data.
    input:
    nameOfScale  = 'standard' (standardize),  'minmax' or 'robustscaler'
    """
    
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
        
    elif nameOfScaler =='minmax':
        scaler = MinMaxScaler()
        
    elif nameOfScaler == 'robustscaler':
        scaler = RobustScaler()

    pipelines = []
    pipelines.append((nameOfScaler+' LinearR'  , 
                      Pipeline([('Scaler', scaler),
                                ('LinearR'   , LinearRegression())])))
    
    pipelines.append((nameOfScaler+' Ridge' , 
                      Pipeline([('Scaler', scaler),('Ridge', Ridge())])))
    
    pipelines.append((nameOfScaler+' Lasso', 
                      Pipeline([('Scaler', scaler),
                                ('Lasso'   , Lasso())])))
    
    pipelines.append((nameOfScaler+' DT' ,
                      Pipeline([('Scaler', scaler),
                                ('DT-R' , DecisionTreeRegressor())])))
    
    #pipelines.append((nameOfScaler+'SVM' ,
         #             Pipeline([('Scaler', scaler),
        #                        ('SVM-R' , SVR(kernel = 'rbf'))])))
    
    pipelines.append((nameOfScaler+' RF'  , 
                      Pipeline([('Scaler', scaler),
                                ('RF-R'  , DecisionTreeRegressor())])))
    return pipelines 

