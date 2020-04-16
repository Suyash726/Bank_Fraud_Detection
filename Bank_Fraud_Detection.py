#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense


# In[2]:


df = pd.read_csv('Desktop/Artificial_Neural_Networks/Churn_Modelling.csv')
df.head()


# In[3]:


df = df.drop(['Surname','RowNumber','CustomerId'],axis=1)


# In[4]:


X = df.drop('Exited',axis=1)
y = df['Exited']

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer

label_encoder_X = LabelEncoder()
X['Geography'] = label_encoder_X.fit_transform(X['Geography'])

label_encoder_g = LabelEncoder()
X['Gender'] = label_encoder_g.fit_transform(X['Gender'])

ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:,1:]


# In[5]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[6]:


from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
X_train = Sc.fit_transform(X_train)
X_test = Sc.fit_transform(X_test)


# In[7]:


classifier = Sequential()
classifier.add(Dense(units=6,init = 'uniform',activation='relu',input_dim = 11))
classifier.add(Dense(units=6,init='uniform',activation='relu'))
classifier.add(Dense(units=6,init='uniform',activation='relu'))
classifier.add(Dense(units=1,init='uniform',activation='sigmoid'))


# In[8]:


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=["accuracy"])


# In[9]:


classifier.fit(X_train,y_train,batch_size=10,epochs=10)


# In[10]:


pred = classifier.predict(X_test)
pred = (pred > 0.5)

from sklearn.metrics import confusion_matrix
ConM = confusion_matrix(pred,y_test)
print(ConM)


# In[ ]:




