#!/usr/bin/env python
# coding: utf-8

# # Libraries to import

# In[1]:


import pandas as pd
import numpy as np

# Load libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


# Function 1: Let's define a function which will instantiate as many models 
#as you wish to

def GetBasedModel():
    basedModels = []
    basedModels.append(('LR'   , LogisticRegression()))
    basedModels.append(('KNN'  , KNeighborsClassifier()))
    basedModels.append(('CART' , DecisionTreeClassifier()))
    basedModels.append(('SVM'  , SVC()))
    basedModels.append(('RF'   , RandomForestClassifier()))
    #basedModels.append(('ET'   , ExtraTreesClassifier())) 
    #basedModels.append(('LDA'  , LinearDiscriminantAnalysis()))
    #basedModels.append(('NB'   , GaussianNB()))
    #basedModels.append(('AB'   , AdaBoostClassifier()))
    #basedModels.append(('GBM'  , GradientBoostingClassifier()))
    return basedModels


# In[ ]:


# Function 2: Let's define a function that will train 
# each individual model described in GetBasedModel() function

def BasedModels(X_train, y_train,scoring, models):
    """
    BasedModels will return the evaluation metric 'auc' after performing
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


# In[ ]:


def MetricsClas(models,X_train, y_train, X_test, y_test):
    for name, model in models:
        print('-*-'*25)
        print('Assessment of ', str(name), '\n')
        model_fit = model.fit(X_train, y_train)
        C_Allmetrics(model_fit, X_train, y_train, X_test, y_test)


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# We will define a function to apply any preprocessing method to the raw data

def GetScaledModel(nameOfScaler):
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
    pipelines.append((nameOfScaler+'LR'  , 
                      Pipeline([('Scaler', scaler),
                                ('LR'  , LogisticRegression())])))
    
    pipelines.append((nameOfScaler+'KNN' , 
                      Pipeline([('Scaler', scaler),('KNN' , 
                                                   KNeighborsClassifier())])))
    pipelines.append((nameOfScaler+'CART', 
                      Pipeline([('Scaler', scaler),
                                ('CART', DecisionTreeClassifier())])))
    pipelines.append((nameOfScaler+'SVM' ,
                      Pipeline([('Scaler', scaler),
                                ('SVM' , SVC(kernel = 'rbf'))])))
    pipelines.append((nameOfScaler+'RF'  , 
                      Pipeline([('Scaler', scaler),
                                ('RF'  , RandomForestClassifier())])))
    
    #pipelines.append((nameOfScaler+'ET'  , Pipeline([('Scaler', scaler),('ET'  , ExtraTreesClassifier())])  ))
    #pipelines.append((nameOfScaler+'LDA' , Pipeline([('Scaler', scaler),('LDA' , LinearDiscriminantAnalysis())])))
    #pipelines.append((nameOfScaler+'NB'  , Pipeline([('Scaler', scaler),('NB'  , GaussianNB())])))
    #pipelines.append((nameOfScaler+'AB'  , Pipeline([('Scaler', scaler),('AB'  , AdaBoostClassifier())])  ))
    #pipelines.append((nameOfScaler+'GBM' , Pipeline([('Scaler', scaler),('GMB' , GradientBoostingClassifier())])  ))

    return pipelines 





