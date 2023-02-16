#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


#reading data from a csv file
poke0= pd.read_csv("data2.csv")


# In[3]:


poke0.shape


# In[4]:


poke0.head()


# In[5]:


#deleting unwanted columns
poke1=poke0.drop(['Unnamed: 0','german_name','japanese_name','egg_type_2','type_2','ability_1','ability_2','ability_hidden','name','species'],axis=1)
poke1.head()


# In[6]:


#checking for null values
poke1.isnull().sum()


# In[7]:


#filling mean value in place of null values
avg_male=np.mean(poke1['percentage_male'])
avg_male


# In[8]:


poke1['percentage_male'].fillna(avg_male,inplace=True)


# In[9]:


#filling median values in place of null values
medCatch=np.mean(poke1['catch_rate'])
poke1['catch_rate'].fillna(medCatch,inplace=True)
medXp= np.mean(poke1['base_experience'])
poke1['base_experience'].fillna(medXp,inplace=True)


# In[10]:


import statistics as st
modeBF=st.mode(poke1['base_friendship'])
poke1['base_friendship'].fillna(modeBF,inplace=True)


# In[11]:


#dropping the rest of null values
poke1.dropna(inplace=True)


# In[12]:


poke1.isnull().sum()


# In[13]:


#plotting a pie chart for growth rate
poke1['growth_rate'].unique()


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

a0= poke1.loc[poke1['growth_rate']=='Medium Slow'].count()[0]
a1= poke1.loc[poke1['growth_rate']=='Medium Fast'].count()[0]
a2= poke1.loc[poke1['growth_rate']=='Fast'].count()[0]
a3= poke1.loc[poke1['growth_rate']=='Slow'].count()[0]
a4= poke1.loc[poke1['growth_rate']=='Fluctuating'].count()[0]
a5= poke1.loc[poke1['growth_rate']=='Erratic'].count()[0]

d=[a0,a1,a2,a3,a4,a5]
l=['Medium Slow', 'Medium Fast', 'Fast', 'Slow', 'Fluctuating','Erratic']
e=[0.1,0.1,0.1,0.1,0.1,0.1]


# In[15]:


plt.pie(d,autopct='%1.2f',labels=l,explode=e)
plt.show()


# In[16]:


#height against speed (scatterplot)
x=poke1['attack'][0:25]
y=poke1['defense'][0:25]


# In[17]:


plt.xlabel("attack")
plt.ylabel("defense")
plt.scatter(x,y)
plt.show()


# In[18]:


#most frequently occuring pokemon type (bargraph)
count= poke1['type_1'].value_counts()
X= count.keys()
Y= count

p=plt.figure()
p.set_figwidth(15)
p.set_figheight(10)
plt.xlabel("number of occurences")
plt.ylabel("pokemon type")
sns.barplot(x=Y,y= X,palette= 'inferno')


# In[19]:


#correlation heatmap
p=plt.figure()
p.set_figwidth(20)
p.set_figheight(20)

co=poke1.corr()
sns.heatmap(co,cmap='GnBu')


# In[20]:


#scatterplot - total points vs base experience
x= poke1['total_points']
y= poke1['base_experience']


# In[21]:


plt.xlabel("total points")
plt.ylabel("base experience")

plt.scatter(x,y)
plt.show()


# In[22]:


p=plt.figure()
p.set_figwidth(20)
p.set_figheight(10)

sns.distplot(poke1['catch_rate'],color='red')


# In[23]:


# hexplot- total points vs hp
sns.jointplot(x=poke1['total_points'],y=poke1['hp'],kind='hex',color='lightcoral',height=10,space=1)


# In[24]:


#formatting the data using label encoder in sklearn
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[25]:


stype1=poke1['type_1'].unique()
type_1_int= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]


# In[26]:


type1=le.fit_transform(poke1['type_1'])


# In[27]:


poke1['type1_int']=type1


# In[28]:


poke1.head()


# In[29]:


growthStr=poke1['growth_rate'].unique()
growthArr=[1,2,3,4,5,6]


# In[30]:


growth=le.fit_transform(poke1['growth_rate'])


# In[31]:


poke1['growth_rate_int']=growth


# In[32]:


poke1.head()


# In[33]:


eggStr=poke1['egg_type_1'].unique()
eggInt=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]


# In[34]:


et=le.fit_transform(poke1['egg_type_1'])


# In[35]:


poke1['egg_type_int']=et


# In[36]:


poke1.head()


# In[37]:


#removing all the string type data columns
poke2=poke1.drop(['type_1','growth_rate','egg_type_1'],axis=1)


# In[38]:


poke2.head()


# In[39]:


#assigning dependent and intependent variables
y=poke2['is_legendary']
x=poke2.drop(['is_sub_legendary','is_legendary','is_mythical'],axis=1)


# In[40]:


#splitting data into test and train set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[41]:


#applying logistic regression 
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()


# In[42]:


log.fit(x_train,y_train)


# In[43]:


Ploglegendary=log.predict(x_test)


# In[44]:


#checking for accuracy score and classification report
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
confusion_matrix(Ploglegendary,y_test)


# In[45]:


accuracy_score(Ploglegendary,y_test)


# In[46]:


classification_report(Ploglegendary,y_test)


# In[47]:


#applying random forest classifier
from sklearn.ensemble import RandomForestClassifier


# In[48]:


r=RandomForestClassifier()


# In[49]:


r.fit(x_train,y_train)
PrfLegendary=r.predict(x_test)


# In[50]:


accuracy_score(PrfLegendary,y_test)


# In[51]:


confusion_matrix(PrfLegendary,y_test)


# In[52]:


classification_report(PrfLegendary,y_test)


# In[53]:


#applying decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[54]:


dt.fit(x_train,y_train)
PdtLegendary =dt.predict(x_test)


# In[55]:


confusion_matrix(PdtLegendary,y_test)


# In[56]:


accuracy_score(PdtLegendary,y_test)


# In[57]:


classification_report(PdtLegendary,y_test)


# In[58]:


import pickle
with open('model3_pkl','wb') as files:
    pickle.dump(r,files)


# In[64]:


poke2[411:455].tail()


# In[ ]:




