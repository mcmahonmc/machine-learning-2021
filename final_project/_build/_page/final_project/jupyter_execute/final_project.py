#!/usr/bin/env python
# coding: utf-8

# # Data-driven assessment of rest-activity patterns
# 
# PSY 394S Machine Learning <br>
# Megan McMahon <br>
# Fall 2021 <br>

# In[1]:


import pandas as pd
import numpy as np
import glob

import matplotlib.pyplot as plt
from matplotlib.dates import WeekdayLocator
import plotly.express as px
import seaborn as sns

from math import ceil
import random

from wearables import preproc

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm


# In[2]:


actig_files = sorted(glob.glob('/Users/mcmahonmc/Box/CogNeuroLab/Aging Decision Making R01/data/actigraphy/raw/*.csv'))

# actig_files_ = random.sample(actig_files, 10)

actig_files[:5]


# In[3]:


# actdf = pd.DataFrame()

# weekday_map= {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu',
#               4:'Fri', 5:'Sat', 6:'Sun'}

# for file in actig_files:
    
#     subject = file.split('raw/')[1][:5]
# #     print(subject)
    
#     # read in actigraphy data
#     act = preproc.preproc(file, 'actiwatch', sr='.5T', truncate=False, write=True, plot=True, recording_period_min=7)
    
#     # find first Monday midnight and start data from there so all subjects starting on same day of the week
#     start = act[(act.index.dayofweek == 1) & (act.index.hour == 0)].index[0]
    
#     # cyclically wrap days that were cut off so not losing data
#     wrap = act[:start]
#     wrap.index = pd.date_range(start=act[start:].last_valid_index() + pd.Timedelta(seconds=30),
#                                   end=act[start:].last_valid_index() + (wrap.last_valid_index() - wrap.first_valid_index()) + pd.Timedelta(seconds=30),
#                                   freq='30S')
    
#     act = pd.concat((act[start:], wrap))
    
#     # keep only seven days of data
#     act = act[act.index <= (act.index[2] + pd.Timedelta(days=7))]
    
#     x_dates = [ weekday_map[day] for day in act.index.dayofweek.unique() ]
    
#     # plot
#     fig, ax = plt.subplots()
#     fig = sns.lineplot(x=act.index, y=act).set(title='sub-%s' % subject)
#     ax.set_xticklabels(labels=x_dates, rotation=45, ha='left')
#     ax.set_xlabel(''); ax.set_ylabel('Activity')
#     plt.show()
    
#     # if subject has < 7 days data, discard, else add to dataset
#     if ( (act.last_valid_index() - act.first_valid_index()) >= pd.Timedelta(days=7) ):
    
#         actdf[subject] = act.values
#         print(actdf.shape)
    
#     else:
        
#         print('sub-%s discarded, recording period %s days' % 
#               (subject, act.last_valid_index() - act.first_valid_index()))


# In[4]:


#sns.heatmap(actdf.isnull(), cmap="YlGnBu")


# In[5]:


# from sklearn.preprocessing import StandardScaler
# x = actdf[:-3].values
# x = StandardScaler().fit_transform(x) # normalizing the features


# In[6]:


# x.shape


# In[7]:


# x


# In[8]:


7 * 24 * 60 * 2


# In[9]:


# np.save('/Users/mcmahonmc/Github/machine-learning-2021/final/actigraphy_data.npy', x)
# actdf.to_csv('/Users/mcmahonmc/Github/machine-learning-2021/final/actigraphy_data_df.csv', index=False)


# ## Load data

# In[10]:


x = np.load('/Users/mcmahonmc/Github/machine-learning-2021/final/actigraphy_data.npy')
print('actigraphy data')
print(x.shape)

actdf = pd.read_csv('/Users/mcmahonmc/Github/machine-learning-2021/final/actigraphy_data_df.csv')
print('actigraphy df')
print(actdf.shape)

targets = pd.read_csv('/Users/mcmahonmc/Github/machine-learning-2021/final/target_data.csv')
print('targets')
print(targets.shape)

targets[:5]


# In[11]:


drop_subs = [ subject for subject in actdf.columns if int(subject) not in targets.subject.values ]
drop_subs_idx = [ actdf.columns.get_loc(subject) for subject in actdf.columns if int(subject) not in targets.subject.values ]

actdf = actdf.drop(drop_subs, axis=1)[:-3]
x = np.delete(x, drop_subs_idx, axis=1)

print(actdf.shape)
print(x.shape)


# In[12]:


from sklearn.preprocessing import StandardScaler
x = actdf.values
x = StandardScaler().fit_transform(x) # normalizing the features
x.shape


# In[13]:


x


# In[14]:


np.mean(x),np.std(x)


# ## Compute traditional rest-activity measures
# 

# In[15]:


# from wearables import fitcosinor, npmetrics
# from datetime import datetime

# rar = pd.DataFrame()

# for subject in actdf.columns:
    
#     df = pd.DataFrame(actdf[subject][:-2]).set_index(pd.to_datetime(
#         pd.date_range(start = pd.to_datetime('2021-01-01 00:00:00'),
#                       end = pd.to_datetime('2021-01-01 00:00:00') + pd.Timedelta(days=7),
#                       freq='30S'),
#         format = '%Y-%m-%d %H:%M:%S'))
    
#     df.columns = ['Activity']
    
#     cr = np.array(fitcosinor.fitcosinor(df)[0].T.values).T[0]
#     nonp = npmetrics.np_metrics_all(df['Activity'])
    
#     rar[subject] = np.concatenate((cr, nonp[:3]))
    
# rar = rar.T
# rar.columns = ['actmin', 'amp', 'alpha', 'beta', 'phi', 'IS', 'IV', 'RA']
# rar


# In[16]:


rar = pd.read_csv('/Users/mcmahonmc/Github/machine-learning-2021/final/rar_df.csv', index_col=0)

drop_subs = [ int(subject) for subject in rar.index if str(subject) not in actdf.columns.values ]
drop_subs

rar2 = (rar[~rar.index.isin(drop_subs)])
print(rar2.shape)


# In[17]:


# rar.to_csv('/Users/mcmahonmc/Github/machine-learning-2021/final/rar_df.csv')


# ### Define targets
# 
# target data uses output from rar dataframe merged with other variables of interest

# **Missing data**
# 
# Missing data here for CESD (this is intended for young adults only), GDS (this is intended for older adults only), and some of the MRI measures (due to poor image quality).
# 
# For targets of interest, will impute missing values with the mean.

# In[18]:


sns.heatmap(targets.isnull(), cmap='terrain')


# In[19]:


# targets.to_csv('/Users/mcmahonmc/Github/machine-learning-2021/final/target_data.csv', index=True)


# # Dimensionality Reduction
# 
# ## PCA

# In[20]:


pca_act = PCA(n_components=2)
principalComponents_act= pca_act.fit_transform(x)


# In[21]:


principal_act_Df = pd.DataFrame(data = principalComponents_act)
principal_act_Df.columns = ['component_' + str(colname + 1) for colname in principal_act_Df.columns]
principal_act_Df


# In[22]:


print('Explained variation per principal component: {}'.format(pca_act.explained_variance_ratio_))
print('Silhouette score: %.3f' % metrics.silhouette_score(pca_act.components_.T, targets['trails_b_group'], metric='euclidean'))


# In[23]:


for col in principal_act_Df.columns:
    
    sns.lineplot(x=principal_act_Df.index, y=principal_act_Df[col])
    plt.show()


# In[24]:


pca_act.components_.T.shape


# In[25]:


sns.scatterplot(x=principal_act_Df['component_1'], y=principal_act_Df['component_2'])


# In[26]:


print('Silhouette score: %.3f' % metrics.silhouette_score(pca_act.components_[[0,1]].T, targets['trails_b_group'], metric='euclidean'))
# print('Silhouette score: %.3f' % metrics.silhouette_score(pca_act.components_[[1,2]].T, targets['trails_b_group'], metric='euclidean'))
# print('Silhouette score: %.3f' % metrics.silhouette_score(pca_act.components_[[0,2]].T, targets['trails_b_group'], metric='euclidean'))


# # Comparison with traditional rest-activity measures

# ## PCA
# ### Component 1
# 
# Shows some correspondance with IS, interdaily stability

# In[27]:


plt.figure(figsize=(10,10))
for col in rar2.columns:
    rarplt = px.scatter(x=pca_act.components_[0],
               y=rar2[col],
               labels = {
                      'x' : 'Component 1',
                      'y' : col
                  },
                  title='PCA Component 1 and %s' % col)
    
    display(rarplt)


# ### Component 2
# 
# Shows some correspondance with acrophase (phi), which is an indicator of rest-activity rhythm phase and corresponds to the modelled time of peak activity

# In[28]:


plt.figure(figsize=(10,10))
for col in rar2.columns:
    rarplt = px.scatter(x=pca_act.components_[1],
               y=rar2[col],
               labels = {
                      'x' : 'Component 2',
                      'y' : col
                  },
                  title='PCA Component 2 and %s' % col)
    
    display(rarplt)


# ## LLE

# In[29]:


plt.figure(figsize=(10,10))
for col in rar2.columns:
    rarplt = px.scatter(x=pca_act.components_[0],
               y=rar2[col],
               labels = {
                      'x' : 'Component 1',
                      'y' : col
                  },
                  title='PCA Component 1 and %s' % col)
    
    display(rarplt)


# # Classification 
# 
# ## SVM
# 
# [Datacamp](https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python)
# 
# ### Cognition

# In[30]:


targets.columns


# In[31]:


targets['trails_b_group'] = np.where(targets['trails_b_z_score_x'] > 1, "High", "Average")
targets['trails_b_group'] = np.where(targets['trails_b_z_score_x'] < -1, "Low", targets['trails_b_group'])

average_idx = targets[targets['trails_b_group'] == 'Average'].index.values

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(np.delete(pca_act.components_[[0,1]].T, average_idx, axis=0), 
                                                    targets['trails_b_group'][targets['trails_b_group'] != 'Average'], 
                                                    test_size=0.3, random_state=100) 

print('Training data points: dim_x = %s, dim_y = %s' % (X_train.shape, y_train.shape))
print('Test data points: dim_x = %s, dim_y = %s' % (X_test.shape, y_test.shape))


# In[32]:


#Create a svm Classifier
kernel_methods = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel_method in kernel_methods:
    
    clf = svm.SVC(kernel=kernel_method)

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print(kernel_method + ", Accuracy: %.2f " % metrics.accuracy_score(y_test, y_pred))


# In[33]:


clf = svm.SVC(kernel='sigmoid')

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[34]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision: %.2f" % metrics.precision_score(y_test, y_pred, average='weighted'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall: %.2f" % metrics.recall_score(y_test, y_pred, average='weighted'))


# In[35]:


pca_act.components_[[0,1]].shape


# In[36]:


markers = y_pred + ', ' + y_test

px.scatter_3d(x=X_test[:,0], 
              y=X_test[:,1], 
              z=targets.iloc[y_test.index.values]['trails_b_z_score_x'],
              color=markers,
              labels = {
                  'x' : 'Component 1',
                  'y' : 'Component 2',
                  'z' : 'Trails B Z-Score',
                  'color': 'Predicted vs. True Performance'
              },
              title='PCA Components and Trails B Z-Score')


# ### Brain

# In[37]:


targets['mod_mean'].fillna(targets['mod_mean'].mean(), inplace=True)
print(targets['mod_mean'])
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
targets['mod_mean_scaled'] = scaler.fit_transform(targets['mod_mean'].values.reshape(-1,1))
targets['mod_mean_scaled'] 

targets['mod_mean_group'] = np.where(targets['mod_mean_scaled'] > 1, "High", "Average")
targets['mod_mean_group'] = np.where(targets['mod_mean_scaled'] < -1, "Low", targets['mod_mean_group'] )

average_idx = targets[targets['mod_mean_group'] == 'Average'].index.values

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(np.delete(pca_act.components_[[0,1]].T, average_idx, axis=0), 
                                                    targets['mod_mean_group'][targets['mod_mean_group'] != 'Average'], 
                                                    test_size=0.3, random_state=100) 

print('Training data points: dim_x = %s, dim_y = %s' % (X_train.shape, y_train.shape))
print('Test data points: dim_x = %s, dim_y = %s' % (X_test.shape, y_test.shape))


# In[38]:


targets['mod_mean_group'].unique()


# In[39]:


#Create a svm Classifier
kernel_methods = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel_method in kernel_methods:
    
    clf = svm.SVC(kernel=kernel_method)

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print(kernel_method + ", Accuracy: %.2f " % metrics.accuracy_score(y_test, y_pred))


# In[40]:


#Create a svm Classifier
# kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
clf = svm.SVC(kernel='sigmoid')

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[41]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision: %.2f" % metrics.precision_score(y_test, y_pred, average='weighted'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall: %.2f" % metrics.recall_score(y_test, y_pred, average='weighted'))


# In[42]:


markers = y_pred + ', ' + y_test

px.scatter_3d(x=X_test[:,0], 
              y=X_test[:,1], 
              z=targets.iloc[y_test.index.values]['mod_mean'],
              color=markers,
              labels = {
                  'x' : 'Component 1',
                  'y' : 'Component 2',
                  'z' : 'DMN-FPN Modularity',
                  'color': 'Predicted vs. Truth'
              },
              title='PCA Components and DMN-FPN Modularity')


# In[43]:


targets['edge_mean'] = targets[[x for x in targets.columns if x.startswith('edge_')]].mean(axis=1)


# In[44]:


targets['edge_mean'].fillna(targets['edge_mean'].mean(), inplace=True)
print(targets['edge_mean'])
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
targets['edge_mean_scaled'] = scaler.fit_transform(targets['edge_mean'].values.reshape(-1,1))
targets['edge_mean_scaled'] 

targets['edge_mean_group'] = np.where(targets['edge_mean_scaled'] > 1, "High", "Average")
targets['edge_mean_group'] = np.where(targets['edge_mean_scaled'] < -1, "Low", targets['edge_mean_group'] )

average_idx = targets[targets['edge_mean_group'] == 'Average'].index.values

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(np.delete(pca_act.components_[[0,1]].T, average_idx, axis=0), 
                                                    targets['edge_mean_group'][targets['edge_mean_group'] != 'Average'], 
                                                    test_size=0.3, random_state=100) 

print('Training data points: dim_x = %s, dim_y = %s' % (X_train.shape, y_train.shape))
print('Test data points: dim_x = %s, dim_y = %s' % (X_test.shape, y_test.shape))


# In[45]:


targets['edge_mean_group'].unique()


# In[46]:


#Create a svm Classifier
kernel_methods = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel_method in kernel_methods:
    
    clf = svm.SVC(kernel=kernel_method)

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print(kernel_method + ", Accuracy: %.2f " % metrics.accuracy_score(y_test, y_pred))


# In[47]:


#Create a svm Classifier
# kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
clf = svm.SVC(kernel='poly')

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[48]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision: %.2f" % metrics.precision_score(y_test, y_pred, average='weighted'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall: %.2f" % metrics.recall_score(y_test, y_pred, average='weighted'))


# In[49]:


markers = y_pred + ', ' + y_test

px.scatter_3d(x=X_test[:,0], 
              y=X_test[:,1], 
              z=targets.iloc[y_test.index.values]['edge_mean'],
              color=markers,
              labels = {
                  'x' : 'Component 1',
                  'y' : 'Component 2',
                  'z' : 'Edge Strength during Retrieval',
                  'color': 'Predicted vs. Truth'
              },
              title='PCA Components and Edge Strength')


# In[ ]:




