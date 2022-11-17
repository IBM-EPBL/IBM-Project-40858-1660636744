#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import joblib


# In[7]:


df=pd.read_csv('Admission_Predict.csv')


# In[8]:


df.head()


# In[9]:


df.describe()


# In[10]:


df.info()


# In[11]:


df.isnull().sum()


# In[12]:


df.head(2)


# In[13]:


df.drop('Serial No.',axis=1,inplace=True)


# In[15]:


df.head(2)


# In[14]:


#Descriptive Statistics


# In[16]:


df.columns


# In[19]:


df[['GRE Score','TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA','Research']].mean()


# In[20]:


df[['University Rating','LOR ']].mode()


# In[21]:


df[['University Rating','LOR ']].median()


# In[22]:


df.head(3)


# In[23]:


df.columns = df.columns.str.replace(' ','')


# In[24]:


df.head(2)


# In[25]:


sns.histplot(df.GREScore)


# In[26]:


sns.histplot(df.TOEFLScore)


# In[29]:


sns.kdeplot(df.UniversityRating,shade=True)


# In[33]:


sns.kdeplot(df.LOR,shade=True)


# In[34]:


sns.kdeplot(df.CGPA,shade=True)


# In[32]:


plt.figure(figsize=(7,5))
df.Research.value_counts().plot(kind='barh')


# In[35]:


plt.figure(figsize=(7,5))
df.UniversityRating.value_counts().plot(kind='barh')


# In[36]:


df.head(2)


# In[40]:


plt.figure(figsize=(20,5))
sns.boxplot(x='GREScore', y='ChanceofAdmit', data=df, palette='rainbow')


# In[41]:


plt.figure(figsize=(20,5))
sns.boxplot(x='TOEFLScore', y='ChanceofAdmit', data=df, palette='rainbow')


# In[43]:


sns.barplot(x='UniversityRating',y='Research',data=df)


# In[45]:


sns.heatmap(df.corr(),annot=True)


# In[46]:


sns.pairplot(df)


# In[47]:


df.head()


# In[102]:


x=df.drop('ChanceofAdmit',axis=1)


# In[103]:


y=df.ChanceofAdmit.values.reshape(-1,1)


# In[106]:


from sklearn.model_selection import train_test_split


# In[107]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


# In[108]:


from sklearn.linear_model import LinearRegression


# In[109]:


lr=LinearRegression()


# In[110]:


lr.fit(x_train,y_train)


# In[111]:


y_pred = lr.predict(x_test)


# In[112]:


from sklearn.metrics import r2_score


# In[113]:


r2_score(y_test,y_pred)


# In[114]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[115]:


mean_absolute_error(y_test,y_pred)


# In[116]:


mean_squared_error(y_test,y_pred)


# In[117]:


from sklearn.ensemble import RandomForestRegressor


# In[118]:


rf=RandomForestRegressor()


# In[119]:


rf.fit(x_train,y_train)


# In[120]:


y_pred=rf.predict(x_test)


# In[121]:


r2_score(y_test,y_pred)


# In[122]:


mean_absolute_error(y_test,y_pred)


# In[123]:


mean_squared_error(y_test,y_pred)


# In[ ]:




