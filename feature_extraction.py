


from readme_parser import rparser
rp = rparser()


# In[3]:


import pandas as pd

df = pd.read_csv('data/m14_merged.csv', index_col=0)


# In[7]:


df.head()


# In[8]:


X = df.drop('label', axis=1).values
y = df['label'].values


# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=10)  
X_train = lda.fit_transform(X_train, y_train)  
X_test = lda.transform(X_test)  


# In[13]:


X_test.shape


# In[14]:


X_train.shape


# In[17]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=4, random_state=0)

classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test)  
y_pred


# In[18]:


from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)  
print(cm)  
print('Accuracy' + str(accuracy_score(y_test, y_pred)))  


# In[ ]:





# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

cdf = pd.concat([df.drop('label', axis=1), pd.get_dummies(df['label'])], axis=1)
corr = cdf.corr()
plt.figure(figsize=(16,10))
sns.heatmap(corr);


# In[23]:


cdf.head()


# In[25]:


df.columns


# In[26]:


feats =   ['BVP_mean', 'BVP_std', 'BVP_min', 'BVP_max',
           'EDA_phasic_mean', 'EDA_phasic_std', 'EDA_phasic_min', 'EDA_phasic_max', 'EDA_smna_mean',
           'EDA_smna_std', 'EDA_smna_min', 'EDA_smna_max', 'EDA_tonic_mean',
           'EDA_tonic_std', 'EDA_tonic_min', 'EDA_tonic_max', 'Resp_mean',
           'Resp_std', 'Resp_min', 'Resp_max', 'TEMP_mean', 'TEMP_std', 'TEMP_min',
           'TEMP_max', 'TEMP_slope', 'BVP_peak_freq', 'age', 'height',
           'weight','subject', 'label']


# In[27]:


df2 = df[feats]#.head()


# In[28]:



cdf = pd.concat([df[feats].drop('label', axis=1), pd.get_dummies(df[feats]['label'])], axis=1)
corr = cdf.corr()
plt.figure(figsize=(16,10))
sns.heatmap(corr);


# In[31]:


corr = cdf.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr[[0,1,2]], annot=True);


# In[ ]:




