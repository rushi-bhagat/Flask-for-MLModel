#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score


# In[3]:


wine = load_wine()


# In[6]:


data = pd.DataFrame(data = np.c_[wine['data'], wine['target']], columns = wine['feature_names'] + ['target'])


# In[7]:


data.head()


# In[8]:


data.describe()


# In[19]:


X_train = data[:-20]
X_test = data[-20:]

y_train = X_train.target
y_test = X_test.target

X_train = X_train.drop('target', 1)
X_test = X_test.drop('target', 1)


# In[20]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)


# In[21]:


y_pred = clf.predict(X_test)


# In[22]:


print("accuracy_score: %.2f" % accuracy_score(y_test, y_pred))


# In[23]:


import pickle
pickle.dump(clf, open('final_prediction.pickle', 'wb'))


# In[ ]:




