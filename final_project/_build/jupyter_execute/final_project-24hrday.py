#!/usr/bin/env python
# coding: utf-8

# # Data-driven assessment of rest-activity patterns
# 
# PSY 394S Machine Learning <br>
# Megan McMahon <br>
# Fall 2021 <br>

# ## Using 24 hour data averaged across hourly bins

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

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics, svm, manifold
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


# ## Load data

# In[2]:


actdf = pd.read_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/actigraphy_data_24hrday_df.csv')
print('actigraphy df')
print(actdf.shape)

targets = pd.read_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/target_data.csv')
targets = targets.set_index('subject')
targets['edge_mean'] = targets[[x for x in targets.columns if x.startswith('edge_')]].mean(axis=1)
print('targets')
print(targets.shape)

targets[:5]


# In[3]:


actdf.T


# In[4]:


x = StandardScaler().fit_transform(actdf.T.values) # normalizing the features


# In[5]:


sns.heatmap(actdf.T, cmap="YlGnBu")


# In[6]:


sns.heatmap(x, cmap="YlGnBu")


# In[7]:


sns.lineplot(x=range(0,len(x[100])), y=x[100])


# In[8]:


drop_subs = [ subject for subject in actdf.columns if int(subject) not in targets.index.values ]
drop_subs_idx = [ actdf.columns.get_loc(subject) for subject in actdf.columns if int(subject) not in targets.index.values ]

actdf = actdf.drop(drop_subs, axis=1)[:-3]
x = np.delete(x, drop_subs_idx, axis=0)

print(actdf.shape)
print(x.shape)


# In[9]:


# from sklearn.preprocessing import StandardScaler
# x = actdf.values
# x = StandardScaler().fit_transform(x) # normalizing the features
# x.shape


# In[10]:


x


# In[11]:


np.mean(x),np.std(x)


# ## Compute traditional rest-activity measures
# 

# In[12]:


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


# In[13]:


rar = pd.read_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/rar_df.csv', index_col=0)

drop_subs = [ int(subject) for subject in rar.index if str(subject) not in actdf.columns.values ]
drop_subs

rar2 = (rar[~rar.index.isin(drop_subs)])
print(rar2.shape)


# In[14]:


[col for col in targets.columns if 'mean_active' in col]


# In[15]:


[col for col in targets.columns if 'mean_sleep' in col]


# In[16]:


rar2 = rar2.merge(targets[['total_ac_mean_active',
                    'duration_mean_sleep', 
                    'total_ac_mean_sleep', 
                   'efficiency_mean_sleep',
                   'sleep_time_mean_sleep',
                   'sleep_time_sd_sleep',
                   'onset_latency_mean_sleep']],
          left_index=True, right_index=True)

sns.heatmap(rar2.isnull(), cmap='terrain')


# In[17]:


# rar.to_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/rar_df.csv')


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


pca = PCA()
data_pcs = pca.fit_transform(x)

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlim(0,20)
print(pca.explained_variance_[:8])


# In[21]:


components_n = 3


# In[22]:


pca = PCA(n_components=components_n)

# X: Xarray-like of shape (n_samples, n_features)
pca.fit(x)
manifold_2Da_pca = pca.fit_transform(x)
manifold_2D_pca = pd.DataFrame(manifold_2Da_pca, 
                               columns=['Component %s' % (i+1) for i in range(0, len(manifold_2Da_pca[0]))])

# Left with 2 dimensions
print(manifold_2D_pca.shape)
manifold_2D_pca.head()


# In[23]:


print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# ## Isomap
# 
# [Benalexkeen resource](https://benalexkeen.com/isomap-for-dimensionality-reduction-in-python/)

# In[24]:


iso = manifold.Isomap(n_neighbors=6, n_components=components_n)
iso.fit(x)
manifold_2Da = iso.transform(x)
manifold_2D_iso = pd.DataFrame(manifold_2Da, 
                               columns=['Component %s' % (i+1) for i in range(0, len(manifold_2Da[0]))])


# Left with 2 dimensions
print(manifold_2D_iso.shape)
manifold_2D_iso.head()


# ## LLE

# In[25]:


lle = manifold.LocallyLinearEmbedding(n_neighbors=6, n_components=components_n)
lle.fit(x)
manifold_2Da_lle = lle.transform(x)
manifold_2D_lle = pd.DataFrame(manifold_2Da, 
                               columns=['Component %s' % (i+1) for i in range(0, len(manifold_2Da[0]))])



# Left with 2 dimensions
print(manifold_2D_lle.shape)
manifold_2D_lle.head()


# # Comparison with traditional sleep and rest-activity measures

# ## PCA
# 
# Component 1 - total activity <br>
# Component 2 - acrophase (phi)

# In[26]:


sns.pairplot(data=manifold_2D_pca.join(rar2.reset_index().drop('index', axis=1)), 
             x_vars = [col for col in manifold_2D_pca.columns if col.startswith('Component')])


# In[27]:


comp1_high_idx, comp1_low_idx = [np.where(manifold_2D_pca['Component 1'] > manifold_2D_pca['Component 1'].median()),
                 np.where(manifold_2D_pca['Component 1'] < manifold_2D_pca['Component 1'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp1_high_idx].mean(axis=0), label='High PC1')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp1_low_idx].mean(axis=0), label='Low PC1')


# In[28]:


comp2_high_idx, comp2_low_idx = [np.where(manifold_2D_pca['Component 2'] > manifold_2D_pca['Component 2'].median()),
                 np.where(manifold_2D_pca['Component 2'] < manifold_2D_pca['Component 2'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp2_high_idx].mean(axis=0), label='High PC2')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp2_low_idx].mean(axis=0), label='Low PC2')


# In[29]:


comp3_high_idx, comp3_low_idx = [np.where(manifold_2D_pca['Component 3'] > manifold_2D_pca['Component 3'].median()),
                 np.where(manifold_2D_pca['Component 3'] < manifold_2D_pca['Component 3'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp3_high_idx].mean(axis=0), label='High PC2')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp3_low_idx].mean(axis=0), label='Low PC2')


# ## Isomap
# 
# Isomap uses the above principle to create a similarity matrix for eigenvalue decomposition. Unlike other non-linear dimensionality reduction like LLE & LPP which only use local information, isomap uses the local information to create a global similarity matrix. The isomap algorithm uses euclidean metrics to prepare the neighborhood graph. Then, it approximates the geodesic distance between two points by measuring shortest path between these points using graph distance. Thus, it approximates both global as well as the local structure of the dataset in the low dimensional embedding. -[Paperspace Blog](https://blog.paperspace.com/dimension-reduction-with-isomap/)
# 
# Component 2 - phi

# In[30]:


sns.pairplot(data=manifold_2D_iso.join(rar2.reset_index().drop('index', axis=1)), 
             x_vars = [col for col in manifold_2D_iso.columns if col.startswith('Component')])


# In[31]:


comp1_high_idx, comp1_low_idx = [np.where(manifold_2D_iso['Component 1'] > manifold_2D_iso['Component 1'].median()),
                 np.where(manifold_2D_iso['Component 1'] < manifold_2D_iso['Component 1'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp1_high_idx].mean(axis=0), label='High C1')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp1_low_idx].mean(axis=0), label='Low C1')


# In[32]:


comp2_high_idx, comp2_low_idx = [np.where(manifold_2D_iso['Component 2'] > manifold_2D_iso['Component 2'].median()),
                 np.where(manifold_2D_iso['Component 2'] < manifold_2D_iso['Component 2'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp2_high_idx].mean(axis=0), label='High C2')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp2_low_idx].mean(axis=0), label='Low C2')


# In[33]:


comp3_high_idx, comp3_low_idx = [np.where(manifold_2D_iso['Component 3'] > manifold_2D_iso['Component 3'].median()),
                 np.where(manifold_2D_iso['Component 3'] < manifold_2D_iso['Component 3'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp3_high_idx].mean(axis=0), label='High C2')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp3_low_idx].mean(axis=0), label='Low C2')


# ## LLE
# 
# Component 2 - phi

# In[34]:


sns.pairplot(data=manifold_2D_lle.join(rar2.reset_index().drop('index', axis=1)), 
             x_vars = [col for col in manifold_2D_lle.columns if col.startswith('Component')])


# In[35]:


comp1_high_idx, comp1_low_idx = [np.where(manifold_2D_lle['Component 1'] > manifold_2D_lle['Component 1'].median()),
                 np.where(manifold_2D_lle['Component 1'] < manifold_2D_lle['Component 1'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp1_high_idx].mean(axis=0), label='High C1')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp1_low_idx].mean(axis=0), label='Low C1')


# In[36]:


comp2_high_idx, comp2_low_idx = [np.where(manifold_2D_lle['Component 2'] > manifold_2D_lle['Component 2'].median()),
                 np.where(manifold_2D_lle['Component 2'] < manifold_2D_lle['Component 2'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp2_high_idx].mean(axis=0), label='High C2')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp2_low_idx].mean(axis=0), label='Low C2')


# In[37]:


comp3_high_idx, comp3_low_idx = [np.where(manifold_2D_lle['Component 3'] > manifold_2D_lle['Component 3'].median()),
                 np.where(manifold_2D_lle['Component 3'] < manifold_2D_lle['Component 3'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp3_high_idx].mean(axis=0), label='High C2')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp3_low_idx].mean(axis=0), label='Low C2')


# In[ ]:





# ## Correlations
# 
# MAE: Average absolute error between the model prediction and the actual observed data. <br>
# RMSE: Lower the RMSE, the more closely a model is able to predict the actual observations.

# In[38]:


sns.pairplot(data=manifold_2D_iso.join(targets[['trails_b_z_score_x', 'mod_mean', 'edge_mean']].reset_index()), 
             x_vars = [col for col in manifold_2D_iso.columns if col.startswith('Component')],
            y_vars = ['trails_b_z_score_x', 'mod_mean', 'edge_mean'])


# In[39]:


#define cross-validation method to use
cv = LeaveOneOut()

#build multiple linear regression model
model = LinearRegression()

#trails b
scores = cross_val_score(model, manifold_2D_iso, targets['trails_b_z_score_x'], 
                         scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

print('Trails B, MAE: %.2f, RMSE: %.2f' % (np.mean(np.absolute(scores)), np.sqrt(np.mean(np.absolute(scores)))))


#modularity
scores = cross_val_score(model, manifold_2D_iso, targets['mod_mean'], 
                         scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

print('Modularity, MAE: %.2f, RMSE: %.2f' % (np.mean(np.absolute(scores)), np.sqrt(np.mean(np.absolute(scores)))))

      
#edge strength
scores = cross_val_score(model, manifold_2D_iso, targets['edge_mean'], 
                         scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)


print('Edge Strength, MAE: %.2f, RMSE: %.2f' % (np.mean(np.absolute(scores)), np.sqrt(np.mean(np.absolute(scores)))))


# # Classification 
# 
# ## SVM
# 
# [Datacamp](https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python)
# 
# ### Cognition

# In[40]:


targets.columns


# In[41]:


targets = targets.reset_index()

targets['trails_b_group'] = np.where(targets['trails_b_z_score_x'] > 1, "High", "Average")
targets['trails_b_group'] = np.where(targets['trails_b_z_score_x'] < -1, "Low", targets['trails_b_group'])

average_idx = targets[targets['trails_b_group'] == 'Average'].index.values


# In[42]:


print('PCA Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_pca, targets['trails_b_group'], metric='euclidean'))
print('Isomap Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_iso, targets['trails_b_group'], metric='euclidean'))
print('LLE Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_lle, targets['trails_b_group'], metric='euclidean'))

silscore = pd.DataFrame({'method': ['PCA', 'Isomap', 'LLE'],
                        'silscore': [metrics.silhouette_score(manifold_2D_pca, targets['trails_b_group'], metric='euclidean'),
                                    metrics.silhouette_score(manifold_2D_iso, targets['trails_b_group'], metric='euclidean'),
                                    metrics.silhouette_score(manifold_2D_lle, targets['trails_b_group'], metric='euclidean')]})


print('\n\n%s' % silscore.min())


# In[43]:


sns.pairplot(manifold_2D_iso.join(targets['trails_b_group'].reset_index()), 
             hue = 'trails_b_group', palette='Set1',
             x_vars = [col for col in manifold_2D_lle.columns if col.startswith('Component')],
             y_vars = [col for col in manifold_2D_lle.columns if col.startswith('Component')])


# :::{note}
# The Silhouette score measures the separability between clusters based on the distances between and within clusters. It calculates the mean intra-cluster distance (a), which is the mean distance within a cluster, and the mean nearest-cluster distance (b), which is the distance between a sample and the nearest cluster it is not a part of, for each sample. Then, the Silhouette coefficient for a sample is (b - a) / max(a, b). - [Maarten Grootendorst](https://www.maartengrootendorst.com/blog/customer/)
# :::
# 
# Isomap yields the lowest silhouette score, suggesting that this dimensionality reduction technique as implemented with the selected parameters outperformed PCA and LLE techniques in terms of cluster separability based on Trails B performance ('high', 'average', 'low').

# In[44]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(manifold_2D_iso.drop(index=average_idx), 
                                                    targets['trails_b_group'][targets['trails_b_group'] != 'Average'], 
                                                    test_size=0.3, random_state=100) 

print('Training data points: dim_x = %s, dim_y = %s' % (X_train.shape, y_train.shape))
print('Test data points: dim_x = %s, dim_y = %s' % (X_test.shape, y_test.shape))


# In[45]:


#Create a svm Classifier
kernel_methods = ['linear', 'poly', 'rbf', 'sigmoid']
accdf = pd.DataFrame()

for kernel_method in kernel_methods:
    
    clf = svm.SVC(kernel=kernel_method)

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    acc = metrics.accuracy_score(y_test, y_pred)
    accdf = accdf.append({'method': kernel_method,
                      'accuracy': acc}, ignore_index=True)
    

    print(kernel_method + ", Accuracy: %.2f " % acc)
    

print('\n\n%s' % accdf.max())


# In[46]:


clf = svm.SVC(kernel=accdf.max()['method'])

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[47]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision: %.2f" % metrics.precision_score(y_test, y_pred, average='weighted'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall: %.2f" % metrics.recall_score(y_test, y_pred, average='weighted'))


# In[48]:


markers = y_pred + ', ' + y_test

px.scatter_3d(x=X_test['Component 1'], 
              y=X_test['Component 2'], 
              z=targets.iloc[y_test.index.values]['trails_b_z_score_x'],
              color=markers,
              labels = {
                  'x' : 'Component 1',
                  'y' : 'Component 2',
                  'z' : 'Trails B Z-Score',
                  'color': 'Predicted vs. True Performance'
              },
              title='Components and Trails B Z-Score')


# ### Brain

# In[49]:


targets['mod_mean'].fillna(targets['mod_mean'].mean(), inplace=True)
print(targets['mod_mean'])
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
targets['mod_mean_scaled'] = scaler.fit_transform(targets['mod_mean'].values.reshape(-1,1))
targets['mod_mean_scaled'] 

targets['mod_mean_group'] = np.where(targets['mod_mean_scaled'] > 1, "High", "Average")
targets['mod_mean_group'] = np.where(targets['mod_mean_scaled'] < -1, "Low", targets['mod_mean_group'] )

average_idx = targets[targets['mod_mean_group'] == 'Average'].index.values


# In[50]:


print('PCA Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_pca, targets['mod_mean_group'], metric='euclidean'))
print('Isomap Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_iso, targets['mod_mean_group'], metric='euclidean'))
print('LLE Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_lle, targets['mod_mean_group'], metric='euclidean'))

silscore = pd.DataFrame({'method': ['PCA', 'Isomap', 'LLE'],
                        'silscore': [metrics.silhouette_score(manifold_2D_pca, targets['mod_mean_group'], metric='euclidean'),
                                    metrics.silhouette_score(manifold_2D_iso, targets['mod_mean_group'], metric='euclidean'),
                                    metrics.silhouette_score(manifold_2D_lle, targets['mod_mean_group'], metric='euclidean')]})


print('\n\n%s' % silscore.min())


# In[51]:


sns.pairplot(manifold_2D_iso.join(targets['mod_mean_group'].reset_index()), 
             hue = 'mod_mean_group', palette='Set1',
             x_vars = [col for col in manifold_2D_iso.columns if col.startswith('Component')],
             y_vars = [col for col in manifold_2D_iso.columns if col.startswith('Component')])


# In[52]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(manifold_2D_iso.drop(index=average_idx),
                                                    targets['mod_mean_group'][targets['mod_mean_group'] != 'Average'], 
                                                    test_size=0.3, random_state=100) 

print('Training data points: dim_x = %s, dim_y = %s' % (X_train.shape, y_train.shape))
print('Test data points: dim_x = %s, dim_y = %s' % (X_test.shape, y_test.shape))


# In[53]:


targets['mod_mean_group'].unique()


# In[54]:


#Create a svm Classifier
kernel_methods = ['linear', 'poly', 'rbf', 'sigmoid']
accdf = pd.DataFrame()

for kernel_method in kernel_methods:
    
    clf = svm.SVC(kernel=kernel_method)

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    acc = metrics.accuracy_score(y_test, y_pred)
    accdf = accdf.append({'method': kernel_method,
                      'accuracy': acc}, ignore_index=True)
    

    print(kernel_method + ", Accuracy: %.2f " % acc)
    

print('\n\n%s' % accdf.max())


# In[55]:


#Create a svm Classifier
# kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
clf = svm.SVC(kernel=accdf.max()['method'])

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[56]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision: %.2f" % metrics.precision_score(y_test, y_pred, average='weighted'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall: %.2f" % metrics.recall_score(y_test, y_pred, average='weighted'))


# In[57]:


markers = y_pred + ', ' + y_test

px.scatter_3d(x=X_test['Component 1'], 
              y=X_test['Component 2'], 
              z=targets.iloc[y_test.index.values]['mod_mean_group'],
              color=markers,
              labels = {
                  'x' : 'Component 1',
                  'y' : 'Component 2',
                  'z' : 'DMN-FPN Modularity',
                  'color': 'Predicted vs. Truth'
              },
              title='Components and DMN-FPN Modularity')


# In[58]:


targets['edge_mean'].fillna(targets['edge_mean'].mean(), inplace=True)
print(targets['edge_mean'])
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
targets['edge_mean_scaled'] = scaler.fit_transform(targets['edge_mean'].values.reshape(-1,1))
targets['edge_mean_scaled'] 

targets['edge_mean_group'] = np.where(targets['edge_mean_scaled'] > 1, "High", "Average")
targets['edge_mean_group'] = np.where(targets['edge_mean_scaled'] < -1, "Low", targets['edge_mean_group'] )

average_idx = targets[targets['edge_mean_group'] == 'Average'].index.values


# In[59]:


print('PCA Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_pca, targets['edge_mean_group'], metric='euclidean'))
print('Isomap Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_iso, targets['edge_mean_group'], metric='euclidean'))
print('LLE Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_lle, targets['edge_mean_group'], metric='euclidean'))

silscore = pd.DataFrame({'method': ['PCA', 'Isomap', 'LLE'],
                        'silscore': [metrics.silhouette_score(manifold_2D_pca, targets['edge_mean_group'], metric='euclidean'),
                                    metrics.silhouette_score(manifold_2D_iso, targets['edge_mean_group'], metric='euclidean'),
                                    metrics.silhouette_score(manifold_2D_lle, targets['edge_mean_group'], metric='euclidean')]})


print('\n\n%s' % silscore.min())


# In[60]:


sns.pairplot(manifold_2D_iso.join(targets['edge_mean_group'].reset_index()), 
             hue = 'edge_mean_group', palette='Set1',
             x_vars = [col for col in manifold_2D_iso.columns if col.startswith('Component')],
             y_vars = [col for col in manifold_2D_iso.columns if col.startswith('Component')])


# In[61]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(manifold_2D_iso.drop(index=average_idx),
                                                    targets['edge_mean_group'][targets['edge_mean_group'] != 'Average'], 
                                                    test_size=0.3, random_state=100) 

print('Training data points: dim_x = %s, dim_y = %s' % (X_train.shape, y_train.shape))
print('Test data points: dim_x = %s, dim_y = %s' % (X_test.shape, y_test.shape))


# In[62]:


targets['edge_mean_group'].unique()


# In[63]:


#Create a svm Classifier
kernel_methods = ['linear', 'poly', 'rbf', 'sigmoid']
accdf = pd.DataFrame()

for kernel_method in kernel_methods:
    
    clf = svm.SVC(kernel=kernel_method)

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    acc = metrics.accuracy_score(y_test, y_pred)
    accdf = accdf.append({'method': kernel_method,
                      'accuracy': acc}, ignore_index=True)
    

    print(kernel_method + ", Accuracy: %.2f " % acc)
    

print('\n\n%s' % accdf.max())


# In[64]:


#Create a svm Classifier
# kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
clf = svm.SVC(kernel=accdf.max()['method'])

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[65]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision: %.2f" % metrics.precision_score(y_test, y_pred, average='weighted'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall: %.2f" % metrics.recall_score(y_test, y_pred, average='weighted'))


# In[66]:


markers = y_pred + ', ' + y_test

px.scatter_3d(x=X_test['Component 1'], 
              y=X_test['Component 2'], 
              z=targets.iloc[y_test.index.values]['edge_mean_group'],
              color=markers,
              labels = {
                  'x' : 'Component 1',
                  'y' : 'Component 2',
                  'z' : 'Memory Network Edge Strength',
                  'color': 'Predicted vs. Truth'
              },
              title='Components and Memory Network Edge Strength')


# In[ ]:




