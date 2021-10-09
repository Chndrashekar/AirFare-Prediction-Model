
# coding: utf-8

# In[13]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

big_df =  data = pd.read_csv('input/processed_df.csv')


# In[14]:


df_train = big_df[0:10683]
df_test = big_df[10683:]
df_test = df_test.drop(['Total_Fare'], axis =1)


# In[15]:


X = df_train.drop(axis=1,columns=['Total_Fare'])
y = df_train.Total_Fare


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# In[17]:


X_test.head()


# In[26]:


import pickle as pkl

stack_gen_model = pkl.load(open('model/stacked_model.pkl', 'rb'))

testval = X_test.iloc[0].to_frame().T
print(testval)


# In[27]:


stack_gen_model.predict(testval)


# In[28]:


print(y_test.iloc[0])

