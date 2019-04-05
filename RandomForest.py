
# coding: utf-8

# In[56]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# In[57]:


data = pd.read_csv("C:\\projects\\coursera\\week5\\1\\abalone.csv")


# In[58]:


data.head()


# In[59]:


data['Sex'][data['Sex']=='M']=1
data['Sex'][data['Sex']=='F']=-1
data['Sex'][data['Sex']=='I']=0


# In[60]:


X= data[data.columns[:-1]]
X
y=data['Rings']


# In[61]:


data['Sex'].head()


# In[62]:


reg = RandomForestRegressor(random_state=1,n_estimators=5)


# In[63]:


reg.fit(X,y)


# In[64]:


from sklearn.cross_validation import KFold


# In[65]:


from sklearn.metrics import r2_score


# In[66]:


pred1 = reg.predict(X)


# In[67]:


print(pred1)


# In[68]:


print(r2_score(y,pred1))


# In[69]:


kf = KFold(n=5,random_state=1,shuffle = True)
for i in range (1,51):
    reg = RandomForestRegressor(random_state=1,n_estimators=i)
    reg.fit(X,y)
    pred1 = reg.predict(X)
    #print(i,'  ',r2_score(y,pred1))


# In[72]:


X_train = []
X_test = []
y_train = []
y_test = []

kf = KFold(4176,n_folds=5,random_state=1,shuffle = True)
for train_index, test_index in kf:

from sklearn.model_selection import cross_val_score

A = cross_val_score(estimator = kf, X, y=None, groups=None, scoring=r2_score, cv=’warn’, n_jobs=None, verbose=0, fit_params=None, pre_dispatch=‘2*n_jobs’, error_score=’raise-deprecating’)

