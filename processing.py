#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab as pl
import pandas as pd
import numpy as np
from sklearn import preprocessing
from preprocess import utils_my

TMP_BUCKET="gs://student_bucket"
data = load_data(TMP_BUCKET)
data = data.fillna(0)
# print("  Train data shape:", data.shape)

# data =pd.read_csv('ALL_FEATURES_Q1+Q2 +Q3+Q4(Pass-fail).csv', low_memory=False, na_values= np.NaN) #FeatureSetTable

data.select_dtypes(include=["object"]).columns

d=[ 'highest_education', 'disability', 'gender','final_result']

for val in d:
    labels,levels = pd.factorize(data[val])
    data[val] = labels
    
    
data=data.drop(['code_module', 'code_presentation','id_student', 'imd_band','age_band', 'region', 'AC T. Clicks',
       'BC T. Clicks', 'AC DataPlus ', 'AC DualPane', 'AC External Quiz',
       'AC Folder', 'AC Forumng', 'AC Glossary', 'AC HomePage',
       'AC HtmlActivity', 'AC Oucollaborate', 'AC Oucontent',
       'AC Ouelluminate', 'AC Ouwiki', 'AC Page', 'AC Questionnaire',
       'AC Quiz', 'AC RepeatActivity', 'AC Resource', 'AC SharedSubPage',
       'AC SubPage', 'AC Url', 'TC_ACTIVITY', 'BC Glossary', 'BC DataPlus',
       'BC DualPane', 'BC ExternalQuiz', 'BC Forumng', 'BC HomePage',
       'BC HtmlActivity', 'BC Oucollaborate', 'BC Oucontent',
       'BC Ouelluminate', 'BC OUwiki', 'BC Page', 'BC Questionnaire',
       'BC Quiz', 'BC Resources', 'BC SharedSubPage', 'BC SubPage', 'BC Url','ModuleAsigns',
       'LateAsignsSub', 'PostA-1', 'PreA-1', 'OnAsClicks', 'gender', 'highest_education', 'num_of_prev_attempts',
       'studied_credits', 'disability'], axis=1)

data=data.fillna(0)


a=(data.iloc[:,1:21])
#a=data
#x = data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(a)
#df = pd.DataFrame(x_scaled,columns=data.columns)
x_scaled


X=x_scaled
X.shape
Y=data['final_result']


X=data.iloc[:,1:21]
Y=data['final_result']
X.shape

