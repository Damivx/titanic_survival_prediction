#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression


# In[9]:


train = pd.read_csv("C:/Users/Damilare Fisayo/Downloads/kaggle_titanic_project/train.csv")
test = pd.read_csv("C:/Users/Damilare Fisayo/Downloads/kaggle_titanic_project/test.csv")


# In[10]:


train.head()
test.head()


# In[15]:


train.shape
test.shape


# In[16]:


train.columns
test.columns


# In[17]:


train.isnull().sum()
test.isnull().sum()


# In[18]:


train_age_mean = train['Age'].mean()


# In[19]:


train['Age'].fillna(train_age_mean, inplace=True)


# In[20]:


test_age_mean = test['Age'].mean()


# In[21]:


test['Age'].fillna(test_age_mean, inplace=True)


# In[24]:


train.isnull().sum()
test.isnull().sum()


# In[26]:


train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)


# In[28]:


train_fare_mean = train['Fare'].mean()


# In[30]:


train['Fare'].fillna(train_fare_mean, inplace=True)


# In[31]:


test_fare_mean = test['Fare'].mean()


# In[33]:


test['Fare'].fillna(test_fare_mean, inplace=True)


# In[34]:


train.isnull().sum()
test.isnull().sum()


# In[36]:


train_age_known = train[train['Age'].notnull()]
train_age_unknown = train[train['Age'].isnull()]


# In[39]:


# Select the input features for training
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X_train = train[features]
y_train = train['Survived']


# In[40]:


# Perform preprocessing steps on the input features
# For example, one-hot encode categorical variables
X_train_encoded = pd.get_dummies(X_train, columns=['Sex', 'Embarked'])


# In[41]:


# Create an instance of the Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train_encoded, y_train)


# In[42]:


# Make predictions on the training set
y_train_pred = model.predict(X_train_encoded)

# Evaluate the performance of the model (example: accuracy)
from sklearn.metrics import accuracy_score

train_accuracy = accuracy_score(y_train, y_train_pred.round())


# In[43]:


train_accuracy


# In[ ]:




